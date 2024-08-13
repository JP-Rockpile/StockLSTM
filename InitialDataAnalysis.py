import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# List of companies
companies = ['BBY', 'LOW', 'WBA', 'KR', 'WMT', 'AMZN', 'TGT', 'HD', 'COST', 'AAPL']

# Date range
start_date = '2017-08-10'
end_date = '2017-09-20'

# Download stock price data for each company
stock_data = {}
for company in companies:
    stock_data[company] = yf.download(company, start=start_date, end=end_date)

# Combine all data into a single DataFrame for closing prices
closing_prices = pd.DataFrame()
for company in companies:
    closing_prices[company] = stock_data[company]['Close']

# Basic analysis: Calculate daily returns
daily_returns = closing_prices.pct_change().dropna()

# Summary statistics for daily returns
summary_stats = daily_returns.describe()

# Print summary statistics
print("Summary Statistics for Daily Returns:")
print(summary_stats)

# Plot closing prices and save the figure
plt.figure(figsize=(14, 7))
for company in companies:
    plt.plot(closing_prices[company], label=company)
plt.title('Stock Closing Prices from August 10th, 2017 to September 20th, 2017')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid()
plt.savefig('BASEclosing_prices.png')  # Save the figure
plt.show()

# Plot daily returns and save the figure
plt.figure(figsize=(14, 7))
for company in companies:
    plt.plot(daily_returns[company], label=company)
plt.title('Daily Returns from August 10th, 2017 to September 20th, 2017')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.grid()
plt.savefig('BASEdaily_returns.png')  # Save the figure
plt.show()
