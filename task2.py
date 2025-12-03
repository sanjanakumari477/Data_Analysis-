import pandas as pd

def gas_storage_contract_value(prices_df, injection_dates, withdrawal_dates,
                               injection_rate, withdrawal_rate,
                               max_storage, storage_cost_per_day):
    # ✅ Fix column names to match your CSV
    if "Dates" in prices_df.columns:
        prices_df = prices_df.rename(columns={"Dates": "Date"})
    if "Prices" in prices_df.columns:
        prices_df = prices_df.rename(columns={"Prices": "Price"})

    # ✅ Convert date column to datetime format
    prices_df['Date'] = pd.to_datetime(prices_df['Date'], errors='coerce')
    prices_df = prices_df.dropna(subset=['Date'])  # remove invalid rows
    prices_df = prices_df.set_index('Date')

    # Initialize variables
    storage_volume = 0
    total_injection_cost = 0
    total_withdrawal_revenue = 0
    total_storage_cost = 0

    # Combine and sort all relevant dates
    all_dates = sorted(list(set(injection_dates + withdrawal_dates)))
    all_dates = pd.to_datetime(all_dates)

    # Loop through all important dates
    for d in all_dates:
        if d not in prices_df.index:
            continue
        price = prices_df.loc[d, 'Price']

        # Injection
        if d in pd.to_datetime(injection_dates):
            volume = min(injection_rate, max_storage - storage_volume)
            total_injection_cost += volume * price
            storage_volume += volume

        # Withdrawal
        if d in pd.to_datetime(withdrawal_dates):
            volume = min(withdrawal_rate, storage_volume)
            total_withdrawal_revenue += volume * price
            storage_volume -= volume

        # Storage cost
        total_storage_cost += storage_volume * storage_cost_per_day

    total_value = total_withdrawal_revenue - total_injection_cost - total_storage_cost
    return {
        "Total Injection Cost": total_injection_cost,
        "Total Withdrawal Revenue": total_withdrawal_revenue,
        "Total Storage Cost": total_storage_cost,
        "Net Contract Value": total_value
    }


# ===================== MAIN SCRIPT =====================

# 🗂️ Load your CSV file — update the filename if needed
file_path = r"C:\Users\sksan\Downloads\Nat_Gas.csv"
df = pd.read_csv(file_path)

# 📆 Define injection and withdrawal dates (must exist in your CSV)
injection_dates = ["2020-11-30", "2020-12-31", "2021-01-31"]
withdrawal_dates = ["2021-02-28", "2021-03-31", "2021-04-30"]

# ⚙️ Run pricing model
result = gas_storage_contract_value(
    prices_df=df,
    injection_dates=injection_dates,
    withdrawal_dates=withdrawal_dates,
    injection_rate=1000,
    withdrawal_rate=1000,
    max_storage=5000,
    storage_cost_per_day=0.05
)

print("\n📊 Contract Pricing Result:")
for k, v in result.items():
    print(f"{k}: {v:.2f}")
