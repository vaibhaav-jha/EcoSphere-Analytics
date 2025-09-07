from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt

key = "2b7914ceb25e5d40f9c9a42007a3b78a"
start_date = "1990-01-01"

fred = Fred(api_key=key)

# Fetch the official GDP series only
data = fred.get_series('GDP', start_date)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['GDP'])
df = df.reset_index()
df.columns = ['Date', 'GDP']

# Convert GDP to billions
df['GDP'] = df['GDP'] / 1000

# Optional: plot
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['GDP'], label='GDP (in billions $)')
plt.xlabel('Date')
plt.ylabel('GDP (Billions USD)')
plt.title('US GDP Over Time')
plt.legend()
plt.grid(True)
plt.show()
