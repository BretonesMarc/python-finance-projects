import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Read CSV data
btc_data = pd.read_csv('BTC-USD.csv')  # read Bitcoin data from CSV file
eth_data = pd.read_csv('ETH-USD.csv')  # read Ethereum data from CSV file
bnb_data = pd.read_csv('BNB-USD.csv')  # read Binance Coin data from CSV file

# Calculate daily returns
btc_data['Return'] = btc_data['Adj Close'].pct_change()  # calculate daily returns for Bitcoin
eth_data['Return'] = eth_data['Adj Close'].pct_change()  # calculate daily returns for Ethereum
bnb_data['Return'] = bnb_data['Adj Close'].pct_change()  # calculate daily returns for Binance Coin

# Remove the first row with NaN values
btc_data = btc_data.iloc[1:]  # remove first row of Bitcoin data
eth_data = eth_data.iloc[1:]  # remove first row of Ethereum data
bnb_data = bnb_data.iloc[1:]  # remove first row of Binance Coin data

# Create a new dataframe to store the daily returns of all assets
portfolio_returns = pd.DataFrame({
    'BTC': btc_data['Return'],
    'ETH': eth_data['Return'],
    'BNB': bnb_data['Return']
})

# Calculate the mean returns and covariance matrix of the assets
mean_returns = portfolio_returns.mean()  # calculate the mean returns of each asset
cov_matrix = portfolio_returns.cov()  # calculate the covariance matrix of the assets

# Function to calculate portfolio returns and volatility
def calculate_portfolio_return_volatility(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights) * 252  # calculate the annualized portfolio return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))  # calculate the annualized portfolio volatility
    return portfolio_return, portfolio_volatility

# Generate random portfolios for brute force optimization
num_portfolios = 50000  # number of random portfolios to generate
random_portfolios = np.zeros((num_portfolios, 6))  # create an empty array to store random portfolios
for i in range(num_portfolios):
    weights = np.random.random(3)  # generate random weights for each asset
    weights /= np.sum(weights)  # normalize the weights to sum up to 1
    portfolio_return, portfolio_volatility = calculate_portfolio_return_volatility(weights, mean_returns, cov_matrix)  # calculate the return and volatility for the portfolio
    sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility  # calculate the Sharpe Ratio for the portfolio
    random_portfolios[i, :3] = [portfolio_volatility, portfolio_return, sharpe_ratio]  # store the portfolio return, volatility, and Sharpe Ratio in the array
    random_portfolios[i, 3:] = weights  #store the weights of each asset in the array

# Find the portfolio with the highest Sharpe Ratio
max_sharpe_idx = np.argmax(random_portfolios[:, 2])  # get the index of the portfolio with the highest Sharpe Ratio
max_sharpe_volatility, max_sharpe_return, max_sharpe_ratio = random_portfolios[max_sharpe_idx, :3]  # get the return, volatility, and Sharpe Ratio

# Get the optimal weights for each coin
optimal_weights = random_portfolios[max_sharpe_idx, 3:]  # Retrieve the asset weights

# Format the weights as percentages
btc_weight_pct = optimal_weights[0] * 100
eth_weight_pct = optimal_weights[1] * 100
bnb_weight_pct = optimal_weights[2] * 100

# Plot the random portfolios and the Efficient Frontier
plt.figure(figsize=(12, 8))
plt.scatter(random_portfolios[:, 0], random_portfolios[:, 1], c=random_portfolios[:, 2], cmap='YlOrRd', marker='o', s=10, alpha=0.8)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='r', s=500, label=f'Optimal Portfolio\nBTC: {btc_weight_pct:.2f}%\nETH: {eth_weight_pct:.2f}%\nBNB: {bnb_weight_pct:.2f}%')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier and Optimal Portfolio')
plt.legend()
plt.show()