#0.1: LIBRARIES
from arch import arch_model 
from math import exp
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns


#0.2: GLOBAL VARS
HORIZON = 5 #minutes
TRADING_DAYS_PER_YEAR = 252
MIN_PER_DAY = 390
ANN_FACTOR = np.sqrt(TRADING_DAYS_PER_YEAR * MIN_PER_DAY / HORIZON )

#1.0: LOAD DATA and DEFINE TARGET VAR
# we wish to find the rolling HORIZON volatility for ONLY trading hours (9:30am --> 4:00pm NY time)
df = pd.read_csv('final.csv').dropna()
df_trading = df[df['market_status'] == 'Trading'].copy()
daily_rolling_sum = df_trading.groupby(df_trading.index.date)['squared_return'].rolling(window=HORIZON).sum().reset_index(level=0, drop=True)
df_trading[f'RVOL_{HORIZON}m'] = np.sqrt(daily_rolling_sum) * ANN_FACTOR
df = df.join(df_trading[f'RVOL_{HORIZON}m'])

#split into training and testing splits. 
split_index = split_index = int(len(df_trading) * 0.7)
df_train = df_trading.iloc[:split_index]
df_test = df_trading.iloc[split_index:]





#2.0: GARCH FORECASTING
def garch_fit_eval(train,test, p, q):
    train['return_scaled'] = train['return'] * 1e05
    model = arch_model(train['return_scaled'], vol='GARCH', p=p,q=q)
    model_fit = model.fit()

    rolling_predictions = []
    test_size = len(test_size)
    return_data = train['return_scaled']

    for i in range(test_size):
        current_data = return_data.iloc[:test_size + i]
        temp_model = arch_model(current_data, vol='GARCH', p=p,q=q)
        forecast = temp_model.forecast(
            params=model_fit.params,
            horizon=1,
            start=current_data.index[-1],
            reindex=False
        )
        rolling_predictions.append(np.sqrt(forecast.variance.iloc[-1,0]))

    test.loc[:,'forecast'] = np.sqrt(rolling_predictions)

    rmse = np.sqrt(np.mean( (test['forecast'] - test[f'RVOL_{HORIZON}m'])**2))


    return [model_fit, rmse, test]




    

# PLOTTING FUNCTION TO COMPARE FORECAST VS HORIZON VOL
def forecast_vs_true_plot(df, model_label:str, title:str ):
    day_to_plot = df.index.date[3]
    plot_data = df.loc[str(day_to_plot)].copy()
    plot_data.index = plot_data.index.tz_convert('America/New_York').tz_localize(None)
    fig, ax = df.subplots(figsize=(15, 8))   
    ax.plot(
    plot_data.index,
    plot_data[f'RVOL_{HORIZON}m'],
    label=f'Actual {HORIZON}-Min Realized Vol',
    color='royalblue',
    linewidth=2
    )

    # Plot the GARCH one-step-ahead forecast
    ax.plot(
        plot_data.index,
        plot_data['forecast'],
        label=model_label,
        color='orangered',
        linestyle='--',
        linewidth=2
    )

    # --- 4. Format the Plot ---
    ax.set_title(f'{title} ({day_to_plot})', fontsize=16)
    ax.set_xlabel('Time (ET)')
    ax.set_ylabel('Annualized Volatility')
    ax.legend()

    formatter = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    p,q = 1,1
    model = garch_fit_eval(df_train, df_test,p,q)
    print(f'=============GARCH({p},{q}) Model Summarry===============\n')
    print(model[0])
    print('===============ERROR METRICS==========\n')
    print(f"rmse: {model[1]}")
    forecast_vs_true_plot(model[2], f'GARCH({p},{q}) Rolling Forecast', 'GARCH Rolling Forecast vs. Actual Volatility' )













