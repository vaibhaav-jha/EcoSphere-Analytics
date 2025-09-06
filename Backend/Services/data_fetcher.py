import yfinance as yf  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

# -------------------
# STEP 1: Fetch Data
# -------------------
ticker = "^GSPC"   # S&P 500 Index
start_date = "1990-01-01"

print("Fetching data from YFinance...")
gspc = yf.download(ticker, start=start_date)

# -------------------
# STEP 2: Feature Engineering
# -------------------

# % Change in Closing Price
gspc["Pct_Change"] = gspc["Close"].pct_change() * 100

# 50-day and 200-day Moving Averages
gspc["MA_50"] = gspc["Close"].rolling(window=50).mean()
gspc["MA_200"] = gspc["Close"].rolling(window=200).mean()

# Average Volume (50-day rolling average) - only if Volume exists
if "Volume" in gspc.columns:
    gspc["Avg_Volume"] = gspc["Volume"].rolling(window=50).mean()
else:
    gspc["Avg_Volume"] = np.nan

# RSI (14-day, using exponential moving averages)
# RSI (14-day, using exponential moving averages)
delta = gspc["Close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)

# flatten to avoid shape (n,1) error
roll_up = pd.Series(gain.flatten(), index=gspc.index).ewm(span=14).mean()
roll_down = pd.Series(loss.flatten(), index=gspc.index).ewm(span=14).mean()

RS = roll_up / roll_down
gspc["RSI"] = 100 - (100 / (1 + RS))


# -------------------
# STEP 3: Clean + Format
# -------------------
# Keep only relevant columns
relevant_data = gspc[["Close", "Pct_Change", "RSI", "MA_50", "MA_200", "Avg_Volume", "Volume"]]

# Drop missing values (from rolling calculations)
relevant_data = relevant_data.dropna()

# Round values for readability
relevant_data = relevant_data.round(2)

# -------------------
# STEP 4: Save to File
# -------------------
relevant_data.to_csv("data_gspc.txt", sep="\t")
print("✅ Week 1 stock analyzer complete. Data saved to data_gspc.txt")

# -------------------
# STEP 5: (Optional Preview & Plot)
# -------------------
print("\nPreview of Cleaned Data:")
print(relevant_data.head(10))

plt.figure(figsize=(12, 6))
plt.plot(relevant_data.index, relevant_data["Close"], label="Closing Price", color="blue")
plt.plot(relevant_data.index, relevant_data["MA_50"], label="50-day MA", color="orange")
plt.plot(relevant_data.index, relevant_data["MA_200"], label="200-day MA", color="red")
plt.title("S&P 500 (^GSPC) Closing Price with 50/200-day MAs (1990-Present)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()
