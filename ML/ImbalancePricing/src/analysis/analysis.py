import pandas as pd
import matplotlib.pyplot as plt

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