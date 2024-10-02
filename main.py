from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
import json
from multiprocessing import cpu_count
import os
import sys
from typing import *
import warnings

try:
    import numpy as np
    import pandas as pd
    import requests
    from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
    from questrade_api import Questrade
except ImportError:
    print("Unable to import required packages.")
    sys.exit()


pd.options.display.float_format = "{:,.2f}".format

# Algorithm config
max_points = 200
tracking_loss_slack = 0.01
tax_loss_slack = 0.01
money_epsilon = 0.05
numerical_epsilon = 1e-8
constraint_power = 6

# Debug config
extra_verbose = False
use_cache = True
use_defaults = False
default_selling_strategy = 4

# Output formatting config
convert_to_units = True
pandas_col_space = 12


def load_config() -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open("config.json", "r") as f:
        config = json.load(f)
    config_accounts = pd.DataFrame(config["accounts"])
    if len(pd.unique(config_accounts["account_number"])) != len(config_accounts["account_number"]):
        print("Duplicate account numbers found in config.json.")
        sys.exit()
    if len(pd.unique(config_accounts["account_label"])) != len(config_accounts["account_label"]):
        print("Duplicate account labels found in config.json.")
        sys.exit()
    config_accounts = config_accounts.sort_values("account_number")
    config_symbols = pd.DataFrame(config["symbols"])
    if len(pd.unique(config_symbols["symbol"])) != len(config_symbols["symbol"]):
        print("Duplicate position symbols found in config.json.")
        sys.exit()
    if config_symbols["weight"].sum() != 1:
        print("Position weights in config don't add to 1.")
        sys.exit()
    config_symbols = config_symbols.sort_values("symbol")
    return config_accounts, config_symbols


def get_questrade_api() -> Questrade:
    print("Getting Questrade API.")
    if os.path.exists(".questrade.json"):
        print("Found saved token.")
        try:
            q = Questrade(token_path=".questrade.json")
            q.time
            assert "time" in q.time
            print("Got Questrade API using saved token.")
            return q
        except:
            print("Failed to get Questrade API using saved token.")
    else:
        print("No saved token was found.")
    refresh_token = input("Please provide a token to proceed: ")
    try:
        q = Questrade(token_path=".questrade.json", refresh_token=refresh_token)
        q.time
        assert "time" in q.time
        print("Got Questrade API using provided token.")
        return q
    except:
        print("Failed to get Questrade API using provided token. Cannot proceed.")
        sys.exit()


def get_exchange_rate() -> dict:
    print("Getting exchange rates.")
    if os.path.exists(".exchange_rate_api.json"):
        print("Found saved token.")
        try:
            with open(".exchange_rate_api.json", "r") as f:
                exchange_rate_api_key = json.load(f)["exchange_rate_api_key"]
            url = f"https://v6.exchangerate-api.com/v6/{exchange_rate_api_key}/latest/USD"
            response = requests.get(url)
            data = response.json()
            assert "conversion_rates" in data
            print("Got exchange rates API using saved token.")
            return data["conversion_rates"]["CAD"]
        except:
            print("Failed to get exchange rates API using saved token.")
    else:
        print("No saved token was found.")
    exchange_rate_api_key = input("Please provide a token to proceed: ")
    try:
        url = f"https://v6.exchangerate-api.com/v6/{exchange_rate_api_key}/latest/USD"
        response = requests.get(url)
        data = response.json()
        assert "conversion_rates" in data
        print("Got exchange rates API using provided token.")
        with open(".exchange_rate_api.json", "w") as f:
            json.dump({"exchange_rate_api_key": exchange_rate_api_key}, f)
        return data["conversion_rates"]["CAD"]
    except:
        print("Failed to get exchange rates API using provided token.")
        return input("Please manually enter exchange rate (1 USD = ??? CAD): ")


def bounds_loss(delta_values_flattened, lb, ub):
    lower_violations = np.maximum(lb - delta_values_flattened, 0)
    upper_violations = np.maximum(delta_values_flattened - ub, 0)
    loss = 0
    loss += np.sum(np.power(lower_violations, constraint_power))
    loss += np.sum(np.power(upper_violations, constraint_power))
    return loss


def contribution_limit_loss(delta_values_flattened, initial_values, account_min_contributions, account_max_contributions, total_contribution):
    delta_values = delta_values_flattened.reshape(initial_values.shape)
    actual_account_contributions = np.sum(delta_values, axis=0)
    actual_total_contribution = np.sum(actual_account_contributions)
    lower_violations = np.maximum(0, account_min_contributions - actual_account_contributions)
    upper_violations = np.maximum(0, actual_account_contributions - account_max_contributions)
    total_violations = np.maximum(0, actual_total_contribution - total_contribution)
    loss = 0
    loss += np.sum(np.power(lower_violations, constraint_power))
    loss += np.sum(np.power(upper_violations, constraint_power))
    loss += np.sum(np.power(total_violations, constraint_power))
    return loss


def tracking_loss(delta_values_flattened, initial_values, symbol_targets):
    delta_values = delta_values_flattened.reshape(initial_values.shape)
    final_values = initial_values + delta_values
    final_allocations = np.sum(final_values, axis=1)
    tracking_errors = final_allocations - symbol_targets
    loss = np.sum(np.square(tracking_errors))
    return loss


def tax_loss(delta_values_flattened, initial_values, taxable_ratios):
    delta_values = delta_values_flattened.reshape(initial_values.shape)
    taxable_capital_gains = -np.minimum(delta_values, 0) * taxable_ratios
    loss = np.sum(taxable_capital_gains)
    return loss


def subtracking_loss(delta_values_flattened, initial_values, symbol_targets_normalized):
    delta_values = delta_values_flattened.reshape(initial_values.shape)
    final_values = initial_values + delta_values
    final_account_values = np.sum(final_values, axis=0)
    account_symbol_targets = final_account_values[None, :] * symbol_targets_normalized[:, None]
    tracking_errors = final_values - account_symbol_targets
    loss = np.sqrt(np.sum(np.square(tracking_errors)))
    return loss


def hybrid_loss(delta_values_flattened, constrained_loss_fns, constraint_values, unconstrained_loss_fn):
    loss_values = np.zeros(shape=len(constrained_loss_fns))
    for i, fn in enumerate(constrained_loss_fns):
        loss_values[i] = fn(delta_values_flattened)
    coeffs = np.zeros(shape=len(constrained_loss_fns))
    nonzero_loss = ~np.isclose(loss_values, 0)
    coeffs[nonzero_loss] = np.maximum(0, 1 - (constraint_values[nonzero_loss] / loss_values[nonzero_loss]))
    return np.sum(loss_values * coeffs) + unconstrained_loss_fn(delta_values_flattened)


def optimize_grid_points(bounds_lb, bounds_ub, max_points, scale_search_iters=5000):
    
    # Ensure all bounds are defined
    assert np.all(np.isfinite(bounds_lb)) and np.all(np.isfinite(bounds_ub))

    # Define function to generate valid grid points
    def generate_grid_points(bounds_lb, bounds_ub, points_per_dim):
        one_dimensional_grids = [np.linspace(lb, ub, num=n+2)[1:-1] for lb, ub, n in zip(bounds_lb, bounds_ub, points_per_dim)]
        return list(product(*one_dimensional_grids))

    # Optimize grid size per dimension
    bound_sizes = np.array([ub - lb for ub, lb in zip(bounds_ub, bounds_lb)])
    scale_lower = 0.0
    scale_upper = np.inf
    scale_guess = 1e-8
    points_per_dim = np.maximum((bound_sizes * scale_guess).astype(int), 1)
    x0s = generate_grid_points(bounds_lb, bounds_ub, points_per_dim)
    best_valid_x0s = []
    best_valid_points_per_dim = None
    for _ in range(scale_search_iters // 2):
        if  len(x0s) >= max_points:
            scale_upper = scale_guess
            scale_guess = (scale_lower + scale_upper) / 2.0
        if  len(x0s) < max_points:
            if len(x0s) > len(best_valid_x0s):
                best_valid_x0s = x0s
                best_valid_points_per_dim = points_per_dim
            if scale_upper == np.inf:
                scale_guess *= 2.0
            else:
                scale_lower = scale_guess
                scale_guess = (scale_lower + scale_upper) / 2.0
        points_per_dim = np.maximum((bound_sizes * scale_guess).astype(int), 1)
        x0s = generate_grid_points(bounds_lb, bounds_ub, points_per_dim)

    # Jiggle the solution a bit to loosen up equal-sized bounds
    rng = np.random.default_rng()
    for _ in range(scale_search_iters // 2):
        jiggle = rng.integers(low=-1, high=2, size=bound_sizes.shape)
        x0s = generate_grid_points(bounds_lb, bounds_ub, best_valid_points_per_dim + jiggle)
        if len(best_valid_x0s) < len(x0s) <= max_points:
            best_valid_x0s = x0s
    if len(best_valid_x0s) == 0:
        print("Scale search failed.")
        sys.exit()
        
    x0s = best_valid_x0s
    return x0s


def grid_search(fun, x0s, bounds, constraints, **kwargs):

    # Perform grid search
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning, message="Singular Jacobian matrix")
        warnings.filterwarnings(action="ignore", category=UserWarning, message="delta_grad == 0.0")
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = []
            for x0 in x0s:
                future = executor.submit(minimize, fun=fun, x0=x0, method="trust-constr", bounds=bounds, constraints=constraints, **kwargs)
                futures.append(future)
            results = [f.result() for f in futures]

    # Find and return best result that satisfies constraints
    best_fun = np.inf
    best_result = None 
    for result in results:
        if result.fun < best_fun:
            valid = True
            if valid:
                best_fun = result.fun
                best_result = result
    return best_result


def round_to_cents(x):
    return np.round(x * 100).astype(int) / 100


def main():

    # =============================================================================
    # API OPERATIONS
    # =============================================================================

    accounts, symbols = load_config()

    if not use_cache:
        q = get_questrade_api()
        exchange_rate = get_exchange_rate()
        with open("exchange_rate.json", "w") as f:
            json.dump({"exchange_rate": exchange_rate}, f)
    else:
        with open("exchange_rate.json", "r") as f:
            exchange_rate = json.load(f)["exchange_rate"]

    # load accounts to check if they're all there
    print("Loading accounts.")
    if not use_cache:
        q_accounts = q.accounts
        with open("accounts.json", "w") as f:
            json.dump(q_accounts, f)
    else:
        with open("accounts.json", "r") as f:
            q_accounts = json.load(f)
    q_accounts = pd.DataFrame(q_accounts["accounts"])
    all_found = True
    for account_number in accounts["account_number"]:
        if account_number not in q_accounts["number"].array:
            print(f"Account number {account_number} isn't in Questrade API data.")
            all_found = False
    if not all_found:
        sys.exit()
    del q_accounts

    # load positions
    print("Loading positions.")
    positions = []
    for account_number in accounts["account_number"]:
        if not use_cache:
            account_positions = q.account_positions(int(account_number))
            with open(f"positions_{account_number}.json", "w") as f:
                json.dump(account_positions, f)
        else:
            with open(f"positions_{account_number}.json", "r") as f:
                account_positions = json.load(f)
        account_positions = pd.DataFrame(account_positions["positions"])
        account_positions["account_number"] = account_number
        positions.append(account_positions)
    positions = pd.concat(positions)
    positions = positions[positions["symbol"].isin(symbols["symbol"])]
    positions = positions[["symbol", "account_number", "currentMarketValue", "openQuantity", "averageEntryPrice"]]
    positions["cost"] = positions["openQuantity"] * positions["averageEntryPrice"]
    del positions["openQuantity"]
    del positions["averageEntryPrice"]
    positions = positions.rename(columns={"currentMarketValue": "initial_value"})
    all_combos = pd.DataFrame(list(product(symbols["symbol"], accounts["account_number"])), columns=["symbol", "account_number"])
    positions = all_combos.merge(positions, on=["symbol", "account_number"], how="left")
    positions = positions.fillna(0)
    
    # collect info about symbols
    if not use_cache:
        symbol_infos = {}
        for symbol in symbols["symbol"]:
            symbol_info = {}
            found = False
            for result in q.symbols_search(prefix=symbol)["symbols"]:
                if result["symbol"] == symbol:
                    symbol_info["symbol_id"] = str(result["symbolId"])
                    symbol_info["currency"] = result["currency"]
                    found = True
                    break
            if not found:
                print(f"Unable to find info on symbol {symbol}.")
                sys.exit()
            result = q.markets_quotes(ids=symbol_info["symbol_id"])["quotes"][0]
            if result["bidPrice"] is None or result["askPrice"] is None:
                print(f"Unable to get current bid/ask prices for symbol {symbol}. Using yesterday's closing price instead.")
                result = q.symbol(id=symbol_info["symbol_id"])["symbols"][0]
                if result["prevDayClosePrice"] is None:
                    print(f"Unable to get yesterday's closing price either.")
                    sys.exit()
                symbol_info["price"] = result["prevDayClosePrice"]
            else:
                symbol_info["price"] = (result["bidPrice"] + result["askPrice"]) / 2
            symbol_infos[symbol] = symbol_info
        with open("symbol_infos.json", "w") as f:
            json.dump(symbol_infos, f)
    else:
        with open("symbol_infos.json", "r") as f:
            symbol_infos = json.load(f)
    symbol_infos = pd.DataFrame(symbol_infos).T
    symbols = symbols.set_index("symbol")
    symbols = symbols.join(symbol_infos)
    symbols = symbols.reset_index()

    # =============================================================================
    # PREPARATION AND USER INPUT
    # =============================================================================

    print("\n", "=" * 80)

    # Sort values
    positions = positions.sort_values(["symbol", "account_number"]).reset_index(drop=True)
    accounts = accounts.sort_values("account_number").reset_index(drop=True)
    symbols = symbols.sort_values("symbol").reset_index(drop=True)

    # Convert USD denominated symbols to CAD
    print(f"\nConverting values to CAD using an exchange rate of {exchange_rate:.3f}CAD = 1USD.")
    is_in_usd = positions.set_index("symbol").join(symbols.set_index("symbol")).reset_index()["currency"] == "USD"
    unconverted_original_values = positions["initial_value"].copy()
    positions.loc[is_in_usd, ["initial_value"]] *= exchange_rate

    # Have user confirm that things look right.
    printable_results = positions.set_index("account_number").join(accounts.set_index("account_number")).reset_index()
    printable_results = printable_results.set_index("symbol").join(symbols.set_index("symbol")).reset_index()
    printable_results["unconverted_original_value"] = unconverted_original_values
    printable_results[["unconverted_original_value", "initial_value"]] = printable_results[["unconverted_original_value", "initial_value"]].apply(round_to_cents)
    for _, account_results in printable_results.groupby("account_number"):
        print(f'\nInitial state for account "{account_results.iloc[0]["account_label"]}":\n')
        account_results = account_results[["symbol", "currency", "unconverted_original_value", "initial_value"]]
        account_results = account_results.rename(columns={"unconverted_original_value": "original_value", "initial_value": "converted_value"})
        print(account_results.to_string(index=False, col_space=pandas_col_space, max_rows=99, max_cols=99, line_width=999))
    print("")
    while True:
        response = input("Does this look right? (y/n) ") if not use_defaults else "y"
        if response.lower() == "n":
            print("Stopping due to user response.")
            sys.exit()
        elif response.lower() == "y":
            break
    print("")

    # Determine if there are capital gains, and if so, if they're taxable
    positions["taxable_ratio"] = (positions["initial_value"] - positions["cost"]) / positions["initial_value"]
    positions.loc[positions["taxable_ratio"] < 0, "taxable_ratio"] = 0
    positions.loc[~positions.set_index("account_number").join(accounts.set_index("account_number")).reset_index()["is_taxable"], "taxable_ratio"] = 0

    # Get user input regarding contributions
    total_contribution = None
    min_contributions = None
    max_contributions = None
    while True:
        while True:
            response = input("Total contribution across all accounts? (blank = zero) $") if not use_defaults else ""
            if len(response) == 0:
                total_contribution = 0
                break
            try:
                total_contribution = float(response)
                break
            except:
                pass
        min_contributions = []
        max_contributions = []
        for _, row in accounts.iterrows():
            if row["is_contributable"]:
                print(f'Contribution limits for account "{row["account_label"]}":')
                while True:
                    min_contribution = None
                    max_contribution = None
                    while True:
                        response = input(f'    Minimum? (blank = zero) $') if not use_defaults else ""
                        if len(response) == 0:
                            min_contribution = 0.0
                            break
                        try:
                            min_contribution = float(response)
                            break
                        except:
                            pass
                    while True:
                        response = input(f'    Maximum? (blank = none) $') if not use_defaults else ""
                        if len(response) == 0:
                            max_contribution = total_contribution
                            break
                        try:
                            max_contribution = float(response)
                            break
                        except:
                            pass
                    if min_contribution <= max_contribution:
                        break
                    else:
                        print("Maximum contribution cannot be less than minimum contribution.")
            else:
                min_contribution = 0.0
                max_contribution = 0.0
            min_contributions.append(min_contribution)
            max_contributions.append(max_contribution)
        if np.sum(min_contributions) <= total_contribution <= np.sum(max_contributions):
            break
        else:
            print("Total contribution cannot be achieved given account contribution constraints.")
    accounts["min_contribution"] = min_contributions
    accounts["max_contribution"] = max_contributions
    print("")

    # Get user input regarding tax strategy
    print("Available tax strategy options:")
    print("    (1) Selling not permitted")
    print("    (2) Selling permitted in non-taxable accounts")
    print("    (3) Selling permitted in non-taxable accounts, and in taxable accounts for capital losses")
    print("    (4) Selling always permitted")
    selling_strategy = None
    while True:
        response = input("Selected tax strategy: ") if not use_defaults else ""
        try:
            if len(response) == 0:
                assert use_defaults
                response = default_selling_strategy
            selling_strategy = int(response)
            assert selling_strategy in (1, 2, 3, 4)
            break
        except:
            pass
    if selling_strategy == 1:
        positions["selling_permitted"] = False
    if selling_strategy == 2:
        positions["selling_permitted"] = ~positions.set_index("account_number").join(accounts.set_index("account_number")).reset_index()["is_taxable"]
    if selling_strategy == 3:
        positions["selling_permitted"] = positions["taxable_ratio"] == 0
    if selling_strategy == 4:
        positions["selling_permitted"] = True
    print("")

    # =============================================================================
    # IT'S TIME TO DO MATH
    # =============================================================================

    print("\n", "=" * 80)

    initial_pivot = positions.pivot(index='symbol', columns='account_number', values='initial_value')
    initial_pivot = initial_pivot.rename(columns={row["account_number"]: row["account_label"] for _, row in accounts.iterrows()})
    initial_pivot = initial_pivot.rename_axis(None, axis="columns")
    initial_values = initial_pivot.to_numpy()
    n_symbols, n_accounts = initial_values.shape

    selling_permitted = positions.pivot(index='symbol', columns='account_number', values='selling_permitted').to_numpy()
    taxable_ratios = positions.pivot(index='symbol', columns='account_number', values='taxable_ratio').to_numpy()

    total_final_value = np.sum(initial_values) + total_contribution
    symbol_targets = symbols["weight"].to_numpy() * total_final_value
    symbol_targets_broadcasted = symbol_targets[:, None] + np.zeros(shape=(1, n_accounts))
    symbol_targets_normalized = symbol_targets / np.sum(symbol_targets)
    account_min_contributions = accounts["min_contribution"].to_numpy()
    account_max_contributions = accounts["max_contribution"].to_numpy()

    def print_diagnostics():
        delta_pivot = initial_pivot.copy()
        delta_pivot.loc[:, :] = delta_values
        delta_pivot = delta_pivot.T
        delta_pivot["contribution"] = np.sum(delta_values, axis=0)
        delta_pivot["min_contribution"] = account_min_contributions
        delta_pivot["max_contribution"] = account_max_contributions
        delta_pivot = delta_pivot.T
        final_pivot = initial_pivot.copy()
        final_pivot.loc[:, :] = final_values
        final_pivot["total"] = np.sum(final_values, axis=1)
        final_pivot["target"] = symbol_targets
        final_pivot["deviation_abs"] = final_pivot["total"] - final_pivot["target"].to_numpy()
        final_pivot["deviation_rel"] = final_pivot["deviation_abs"] / symbol_targets
        print("\nDelta values:")
        print(delta_pivot.to_string(max_rows=99, max_cols=99, line_width=999))
        print("\nFinal values:")
        print(final_pivot.to_string(max_rows=99, max_cols=99, line_width=999))
        print("")
        print("Hybrid loss components:")
        for name, fn in zip(hybrid_loss_fn_names, hybrid_loss_constrained_loss_fns):
            print(f"    {name}: {fn(delta_values.flatten())}")
        print(f"    {unconstrained_loss_fn_name}: {unconstrained_loss_fn(delta_values.flatten())}")
        print("")

    # Initialize constraints
    bounds = None
    constraints = []
    hybrid_loss_constrained_loss_fns = []
    hybrid_loss_constraint_values = []
    hybrid_loss_fn_names = []
    unconstrained_loss_fn = None
    unconstrained_loss_fn_name = None
    
    # Selling-based constraints
    bounds_lb = []
    bounds_ub = []
    for i, j in product(range(n_symbols), range(n_accounts)):
        if not selling_permitted[i, j]:
            # Selling not permitted
            bounds_lb.append(0)
        else:
            # Selling permitted, but final value must be non-negative
            bounds_lb.append(-1 * initial_values[i, j])
        bounds_ub.append(account_max_contributions[j] + np.sum(initial_values[:, j]) - initial_values[i, j])
    bounds_lb = np.array(bounds_lb)
    bounds_ub = np.array(bounds_ub)
    bounds_middle = (bounds_lb + bounds_ub) / 2
    bounds_lb = np.minimum(bounds_lb, bounds_middle - (numerical_epsilon / 2))
    bounds_ub = np.maximum(bounds_ub, bounds_middle + (numerical_epsilon / 2))
    print(f"\nFinding grid points to initialize optimizations.")
    x0s = optimize_grid_points(bounds_lb, bounds_ub, max_points=max_points)
    print(f"Using {len(x0s)} grid points.")

    bounds = Bounds(lb=bounds_lb, ub=bounds_ub)
    bounds_loss_partial_fn = partial(bounds_loss, lb=np.array(bounds_lb), ub=np.array(bounds_ub))
    hybrid_loss_constrained_loss_fns.append(bounds_loss_partial_fn)
    hybrid_loss_constraint_values.append(0.0)
    hybrid_loss_fn_names.append("bounds_loss")

    # Linear constraint for total and per-account contributions
    linear_constraint_lb = []
    linear_constraint_ub = []
    linear_constraint_A = []
    linear_constraint_lb.append(total_contribution)
    linear_constraint_ub.append(total_contribution)
    linear_constraint_A.append(np.ones(shape=(n_symbols, n_accounts)).flatten())
    for i, (min_contribution, max_contribution) in enumerate(zip(min_contributions, max_contributions)):
        linear_constraint_lb.append(min_contribution)
        linear_constraint_ub.append(max_contribution)
        A_row = np.zeros(shape=(n_symbols, n_accounts))
        A_row[:, i] = 1
        linear_constraint_A.append(A_row.flatten())
    linear_constraint_lb = np.array(linear_constraint_lb)
    linear_constraint_ub = np.array(linear_constraint_ub)
    linear_constraint_A = np.array(linear_constraint_A)
    linear_constraint_middle = (linear_constraint_lb + linear_constraint_ub) / 2
    linear_constraint_lb = np.minimum(linear_constraint_lb, linear_constraint_middle - (numerical_epsilon / 2))
    linear_constraint_ub = np.maximum(linear_constraint_ub, linear_constraint_middle + (numerical_epsilon / 2))
    contribution_constraint = LinearConstraint(A=linear_constraint_A, lb=linear_constraint_lb, ub=linear_constraint_ub)
    constraints.append(contribution_constraint)
    contribution_limit_loss_partial_fn = partial(contribution_limit_loss, initial_values=initial_values, account_min_contributions=account_min_contributions, account_max_contributions=account_max_contributions, total_contribution=total_contribution)
    hybrid_loss_constrained_loss_fns.append(contribution_limit_loss_partial_fn)
    hybrid_loss_constraint_values.append(0.0)
    hybrid_loss_fn_names.append("contribution_limit_loss")

    # Optimize tracking loss
    print(f"Running optimization to minimize tracking error.")
    tracking_loss_partial_fn = partial(tracking_loss, initial_values=initial_values, symbol_targets=symbol_targets)
    unconstrained_loss_fn = tracking_loss_partial_fn
    unconstrained_loss_fn_name = "tracking_loss"
    hybrid_loss_partial_fn = partial(hybrid_loss, constrained_loss_fns=hybrid_loss_constrained_loss_fns, constraint_values=np.array(hybrid_loss_constraint_values), unconstrained_loss_fn=unconstrained_loss_fn)
    result = grid_search(fun=hybrid_loss_partial_fn, x0s=x0s, bounds=bounds, constraints=constraints)
    delta_values = result.x.reshape(initial_values.shape)
    final_values = initial_values + delta_values
    tracking_errors_interpretable = np.abs((np.sum(final_values, axis=1) - symbol_targets)) / symbol_targets
    tracking_error_interpretable = np.average(tracking_errors_interpretable, weights=symbol_targets)
    print(f"Achieved tracking error of {(tracking_error_interpretable * 100):.2f}%.")

    if extra_verbose:
        print_diagnostics()

    # Construct tracking loss constraint
    max_tracking_loss = np.sum(np.square(np.abs(np.sum(initial_values + delta_values, axis=1) - symbol_targets) + (tracking_loss_slack * symbol_targets / n_symbols)))
    tracking_loss_constraint = NonlinearConstraint(fun=tracking_loss_partial_fn, lb=-np.inf, ub=max_tracking_loss)
    constraints.append(tracking_loss_constraint)
    hybrid_loss_constrained_loss_fns.append(tracking_loss_partial_fn)
    hybrid_loss_constraint_values.append(max_tracking_loss)
    hybrid_loss_fn_names.append("tracking_loss")

    # Optimize tax loss if taxable sales are allowed
    tax_loss_partial_fn = partial(tax_loss, initial_values=initial_values, taxable_ratios=taxable_ratios)
    prev_result = result
    prev_taxable_gains = tax_loss_partial_fn(prev_result.x)
    if prev_taxable_gains > money_epsilon:
        print(f"Current solution incurs ${prev_taxable_gains:.2f} of taxable capital gains.")
        print(f"Running optimization to minimize taxable capital gains.")
        unconstrained_loss_fn = tax_loss_partial_fn
        unconstrained_loss_fn_name = "tax_loss"
        hybrid_loss_partial_fn = partial(hybrid_loss, constrained_loss_fns=hybrid_loss_constrained_loss_fns, constraint_values=np.array(hybrid_loss_constraint_values), unconstrained_loss_fn=unconstrained_loss_fn)
        result = grid_search(fun=hybrid_loss_partial_fn, x0s=x0s, bounds=bounds, constraints=constraints)
        taxable_gains = tax_loss_partial_fn(result.x)
        if taxable_gains < prev_taxable_gains:
            print(f"Achieved taxable capital gains of ${(taxable_gains):.2f}.")
            delta_values = result.x.reshape(initial_values.shape)
            final_values = initial_values + delta_values
            tracking_errors_interpretable = np.abs((np.sum(final_values, axis=1) - symbol_targets)) / symbol_targets
            tracking_error_interpretable = np.average(tracking_errors_interpretable, weights=symbol_targets)
            optimal_taxable_gains = taxable_gains
        else:
            print(f"Unable to reduce capital gains.")
            result = prev_result
            optimal_taxable_gains = prev_taxable_gains

        if extra_verbose:
            print_diagnostics()

        # Construct tax loss constraint
        max_taxable_gains = optimal_taxable_gains * (1 + tax_loss_slack)
        tax_loss_constraint = NonlinearConstraint(fun=tax_loss_partial_fn, lb=-np.inf, ub=max_taxable_gains)
        constraints.append(tax_loss_constraint)
        hybrid_loss_constrained_loss_fns.append(tax_loss_partial_fn)
        hybrid_loss_constraint_values.append(max_taxable_gains)
        hybrid_loss_fn_names.append("tax_loss")

    # Optimize tracking error within individual accounts
    subtracking_errors_interpretable = np.abs((final_values / np.sum(final_values, axis=0)) - symbol_targets_normalized[:, None])
    subtracking_error_interpretable = np.average(subtracking_errors_interpretable, weights=symbol_targets_broadcasted)
    print(f"Current solution has account tracking error of {(subtracking_error_interpretable * 100):.2f}%.")
    print(f"Running optimization to minimize account tracking error.")
    subtracking_loss_partial_fn = partial(subtracking_loss, initial_values=initial_values, symbol_targets_normalized=symbol_targets_normalized)
    unconstrained_loss_fn = subtracking_loss_partial_fn
    unconstrained_loss_fn_name = "subtracking_loss"
    hybrid_loss_partial_fn = partial(hybrid_loss, constrained_loss_fns=hybrid_loss_constrained_loss_fns, constraint_values=np.array(hybrid_loss_constraint_values), unconstrained_loss_fn=unconstrained_loss_fn)
    prev_result = result
    prev_subtracking_error = subtracking_loss_partial_fn(prev_result.x)
    result = grid_search(fun=hybrid_loss_partial_fn, x0s=x0s, bounds=bounds, constraints=constraints)
    subtracking_error = subtracking_loss_partial_fn(result.x)
    if subtracking_error < prev_subtracking_error:
        delta_values = result.x.reshape(initial_values.shape)
        final_values = initial_values + delta_values
        subtracking_errors_interpretable = np.abs((final_values / np.sum(final_values, axis=0)) - symbol_targets_normalized[:, None])
        subtracking_error_interpretable = np.average(subtracking_errors_interpretable, weights=symbol_targets_broadcasted)
        print(f"Achieved account tracking error of {(subtracking_error_interpretable * 100):.2f}%.")
    else:
        print(f"Unable to reduce account tracking error.")
        result = prev_result

    if extra_verbose:
        print_diagnostics()

    # =============================================================================
    # OUTPUT
    # =============================================================================

    print("\n", "=" * 80)

    # Print solution summary
    delta_values = result.x.reshape(initial_values.shape)
    final_values = initial_values + delta_values
    tracking_errors_interpretable = np.abs((np.sum(final_values, axis=1) - symbol_targets)) / symbol_targets
    tracking_error_interpretable = np.average(tracking_errors_interpretable, weights=symbol_targets)
    subtracking_errors_interpretable = np.abs((final_values / np.sum(final_values, axis=0)) - symbol_targets_normalized[:, None])
    subtracking_error_interpretable = np.average(subtracking_errors_interpretable, weights=symbol_targets_broadcasted)
    taxable_gains = tax_loss_partial_fn(delta_values.flatten())
    print(f"\nSolution summary: ")
    print(f"    Global tracking error  : {(tracking_error_interpretable * 100):.2f}%")
    print(f"    Account tracking error : {(subtracking_error_interpretable * 100):.2f}%")
    print(f"    Taxable capital gains  : ${taxable_gains:.2f}")

    if extra_verbose:
        print_diagnostics()

    # Double check constraints
    print("")
    constraints_violated = False
    actual_total_contribution = np.sum(final_values) - np.sum(initial_values)
    total_contribution_error = actual_total_contribution - total_contribution
    if np.abs(total_contribution_error) > money_epsilon:
        print(f"Violated constraint found! Total contribution error of ${total_contribution_error:.2f}.")
    n_negative_final_values = np.sum(np.round(final_values * 100).astype(int) < 0).astype(int)
    if n_negative_final_values > money_epsilon:
        print(f"Violated constraint found! {n_negative_final_values} positions have negative final values.")
        constraints_violated = True
    account_contributions = np.sum(delta_values, axis=0)
    for i in range(n_accounts):
        row = accounts.iloc[i]
        min_violation = row["min_contribution"] - account_contributions[i]
        if min_violation > money_epsilon:
            print(f"Violated constraint found! {row['account_label']} contribution minimum violated by ${min_violation:.2f}.")
            constraints_violated = True
        max_violation = account_contributions[i] - row["max_contribution"]
        if max_violation > money_epsilon:
            print(f"Violated constraint found! {row['account_label']} contribution maximum violated by ${max_violation:.2f}.")
            constraints_violated = True
    if not constraints_violated:
        print("No violated constraints found.")
    
    # Ask for user to confirm
    print("")
    while True:
        response = input("Does this look right? (y/n) ") if not use_defaults else "y"
        if response.lower() == "n":
            print("Stopping due to user response.")
            sys.exit()
        elif response.lower() == "y":
            break

    # Update dataframes
    print("\n", "=" * 80)
    positions["delta_value"] = delta_values.flatten()
    positions["final_value"] = final_values.flatten()
    symbols["initial_weight"] = np.sum(initial_values, axis=1) / np.sum(initial_values)
    symbols["initial_tracking_error"] = symbols["initial_weight"] - symbol_targets_normalized
    symbols["final_weight"] = np.sum(final_values, axis=1) / np.sum(final_values)
    symbols["final_tracking_error"] = symbols["final_weight"] - symbol_targets_normalized
    accounts["contribution"] = np.sum(delta_values, axis=0)
    positions["initial_subweight"] = (initial_values / np.sum(initial_values, axis=0)).flatten()
    positions["initial_subtracking_error"] = ((initial_values / np.sum(initial_values, axis=0)) - symbol_targets_normalized[:, None]).flatten()
    positions["final_subweight"] = (final_values / np.sum(final_values, axis=0)).flatten()
    positions["final_subtracking_error"] = ((final_values / np.sum(final_values, axis=0)) - symbol_targets_normalized[:, None]).flatten()
    positions.loc[is_in_usd, ["initial_value", "delta_value", "final_value"]] /= exchange_rate

    # Sort positions to be more printable
    positions = positions.set_index(["account_number", "symbol"]).sort_index().reset_index()

    # Print user-facing info
    printable_results = positions
    printable_results = printable_results.set_index("symbol").join(symbols.set_index("symbol")).reset_index()
    printable_results = printable_results.set_index("account_number").join(accounts.set_index("account_number")).reset_index()
    printable_results[["initial_value", "final_value", "delta_value"]] = printable_results[["initial_value", "final_value", "delta_value"]].apply(round_to_cents)
    if convert_to_units:
        printable_results["delta_units"] = printable_results["delta_value"] / printable_results["price"]
    print("\nFinal optimization results:")
    for account_number, account_results in printable_results.groupby("account_number"):
        account_contribution = accounts.set_index("account_number").loc[account_number]["contribution"]
        account_contribution_printable = f"${np.abs(account_contribution):.2f}"
        if np.round(account_contribution * 100).astype(int) < 0:
            account_contribution_printable = "-" + account_contribution_printable
        print(f'\nTotal contribution to account "{account_results.iloc[0]["account_label"]}": {account_contribution_printable}\n')
        if convert_to_units:
            account_results = account_results[["symbol", "currency", "initial_value", "final_value", "delta_value", "delta_units", "symbol"]]
        else:
            account_results = account_results[["symbol", "currency", "initial_value", "final_value", "delta_value", "symbol"]]
        print(account_results.to_string(index=False, col_space=pandas_col_space, max_rows=99, max_cols=99, line_width=999))

    # Exit
    print("\n", "=" * 80)
    print("")
    while True:
        response = input("Print additional information before exiting? (y/n | empty = n) ") if not use_defaults else ""
        if len(response) == 0:
            response = "n"
        if response.lower() == "n":
            break
        elif response.lower() == "y":
            print("\nSymbols:")
            print(symbols.to_string(max_rows=99, max_cols=99, line_width=999))
            print("\nAccounts:")
            print(accounts.to_string(max_rows=99, max_cols=99, line_width=999))
            print("\nPositions:")
            print(positions.to_string(max_rows=99, max_cols=99, line_width=999))
            print_diagnostics()
            break
    print("See you next time.\n")
    sys.exit()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()