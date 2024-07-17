import pandas as pd
import matplotlib.pyplot as plt

generation_df = pd.read_csv('ML\\ImbalancePricing\\data\\GenerationByFuelType.csv')
demand_df = pd.read_csv('ML\\ImbalancePricing\\data\\RollingSystemDemand.csv')
prices_df = pd.read_csv('ML\\ImbalancePricing\\data\\SystemSellAndBuyPrices.csv')

generation_df['StartTime'] = pd.to_datetime(generation_df['StartTime'])
demand_df['StartTime'] = pd.to_datetime(demand_df['StartTime'])
prices_df['StartTime'] = pd.to_datetime(prices_df['StartTime'])

total_generation = generation_df.groupby('StartTime')['Generation'].sum().reset_index()

merged_df = pd.merge(total_generation, demand_df, on='StartTime', how='inner')

final_df = pd.merge(merged_df, prices_df[['StartTime', 'SystemSellPrice', 'SystemBuyPrice']], on='StartTime', how='inner')

fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(final_df['StartTime'], final_df['Generation'], label='Total Generation', color='b')
ax1.plot(final_df['StartTime'], final_df['Demand'], label='Total Demand', color='g')
ax1.set_xlabel('Time')
ax1.set_ylabel('MW')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(final_df['StartTime'], final_df['SystemSellPrice'], label='System Sell / Buy Price', color='orange')
ax2.set_ylabel('Price')
ax2.legend(loc='upper right')

plt.title('Total Generation, Total Demand, and System Sell Price')
plt.show()

train_1_df = pd.read_csv("ML\\ImbalancePricing\\data\\test\\test_1.csv")
train_1_df['SettlementDate'] = pd.to_datetime(train_1_df['SettlementDate'])
train_1_df['StartTime'] = pd.to_datetime(train_1_df['StartTime'])
fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

axs[0].plot(train_1_df['StartTime'], train_1_df['SystemSellPrice'], label='System Sell Price', color='red')
axs[0].plot(train_1_df['StartTime'], train_1_df['SystemBuyPrice'], label='System Buy Price', color='blue')
axs[0].set_title('System Sell and Buy Prices Over Time (Train 1)')
axs[0].set_ylabel('£/MWh')
axs[0].legend()

axs[1].plot(train_1_df['StartTime'], train_1_df['NetImbalanceVolume'], label='Net Imbalance Volume', color='green')
axs[1].set_title('Net Imbalance Volume Over Time (Train 1)')
axs[1].set_ylabel('MWh')
axs[1].legend()

axs[2].plot(train_1_df['StartTime'], train_1_df['TotalAcceptedOfferVolume'], label='Total Accepted Offer Volume', color='purple')
axs[2].plot(train_1_df['StartTime'], train_1_df['TotalAcceptedBidVolume'], label='Total Accepted Bid Volume', color='orange')
axs[2].set_title('Total Accepted Offer and Bid Volumes Over Time (Train 1)')
axs[2].set_ylabel('MWh')
axs[2].legend()

axs[2].set_xlabel('Time')

plt.xticks(rotation=45)
plt.show()