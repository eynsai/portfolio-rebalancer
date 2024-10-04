# portfolio-rebalancer

This script helps you rebalance investment portfolios. 

It can handle:
- Rebalancing across multiple accounts
- Constraints on per-account and total contributions
- Optimization of taxable capital gains

Currently, it only supports the use of the Questrade API. I'd like to refactor the script to be compatible with multiple brokerage APIs and data input methods via interchangable backends.

## Usage

### Linux

`./start.sh` will create and activate the necessary python virtual environment, then run the calculator.

### Windows PowerShell

`.\start.ps1` will create and activate the necessary python virtual environment, then run the calculator.

You may need to first relax PowerShell script execution restrictions using `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. 
Be aware that this will reduced in a reduced security level (but it's probably safe if you know what you're doing).
