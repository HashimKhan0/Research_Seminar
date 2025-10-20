'''
Volatility measures the degree of variation in asset prices over time. 
In finance, it represents uncertainty or risk: higher volatility = larger potential price swings. 
For USO (which tracks crude oil futures), volatility reflects energy market uncertainty, geopolitical risk, and liquidity during trading hours.
'''


'''
FORMS OF VOL

instantaneous vol -> variance of log returns per unit time
realized vol --> square root of intrada squared returns
Rolling vol --> time varying std deviation ove a moving window
Implied volatiltiy --> expected future vol from options prices
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


WINDOW = 60 #minutes


df = pd.read_csv('data.csv')

df[f'rolling_vol({WINDOW})'] = df['return'].rolling(WINDOW).std() * np.sqrt(WINDOW)
df['realized_vol'] = df['returns']**2
df['cumulative_realized_vol'] = df['realized_vol'].rolling(WINDOW).sum()*0.5



fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
ax[0].plot(df['returns'], color='gray', alpha=0.6)
ax[0].set_title('Minute Returns')

ax[1].plot(df['rolling_vol'], color='blue')
ax[1].set_title('Rolling (60-min) Volatility')

ax[2].plot(df['cumulative_realized_vol'], color='orange')
ax[2].set_title('Realized Volatility (Windowed)')

plt.tight_layout()
plt.show()