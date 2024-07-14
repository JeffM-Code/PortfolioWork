import pandas as pd
import matplotlib.pyplot as plt

test_df = pd.read_csv('ML\\GasTurbineConsumption\\data\\test\\test_1.csv')

# Input Voltage vs. Time
plt.figure(figsize=(12, 6))
plt.plot(test_df['time'], test_df['input_voltage'], color='blue', label='Input Voltage')
plt.xlabel('Time [sec]')
plt.ylabel('Input Voltage [V]')
plt.title('Input Voltage vs. Time')
plt.legend()
plt.grid(True)
plt.show()

# Electrical Power vs. Time
plt.figure(figsize=(12, 6))
plt.plot(test_df['time'], test_df['el_power'], color='green', label='Electrical Power')
plt.xlabel('Time [sec]')
plt.ylabel('Electrical Power [W]')
plt.title('Electrical Power vs. Time')
plt.legend()
plt.grid(True)
plt.show()