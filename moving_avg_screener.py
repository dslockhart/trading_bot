import datetime as dt
import numpy as np
import pandas as pd
from itertools import product
from pylab import plt, mpl
from typing import List, Callable, Dict, Any, Tuple
from itertools import product
from pandas_datareader import data as pdr
import yfinance as yf
from oanda_candles import CandleClient, Pair, Gran, Candle
from oandapyV20 import API
from oandapyV20.contrib.requests import MarketOrderRequest
from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails
import oandapyV20.endpoints.orders as orders

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
plt.style.use("seaborn-v0_8")
mpl.rcParams["font.family"] = "serif"
yf.pdr_override()
access_token = "35a600dee1346b9d7454ab12fc7d899e-9b9dbcb7d1e9bf43ebfca1849ea0602b"
account_id = "101-004-27267438-001"


def get_data(currency: str) -> pd.DataFrame:
    start = dt.datetime.now() - dt.timedelta(days=365 * 3)
    end = dt.datetime.now()
    df = pdr.get_data_yahoo(currency, start, end)

    return df.iloc[100:]


def strategy_sma_9_61(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA_30"] = df["Close"].rolling(window=9).mean()
    df["SMA_100"] = df["Close"].rolling(window=61).mean()
    df["Signal"] = np.where(df["SMA_30"] > df["SMA_100"], 1, -1)
    return df


def strategy_red_white_blue(df: pd.DataFrame) -> pd.DataFrame:
    emasUsed = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]
    for ema in emasUsed:
        df[f'Ema_{str(ema)}'] = df['Adj Close'].ewm(span=ema, adjust=False).mean()
    cond = [df.loc[:, 'Ema_3':'Ema_15'].min(axis='columns') > df.loc[:, 'Ema_30':'Ema_60'].max(axis='columns')]
    output = [1]
    df['Signal'] = np.select(cond, output, -1)
    return df


def get_signals(df: pd.DataFrame, strategy: Callable) -> pd.DataFrame:
    df = strategy(df)
    df = df.iloc[100:]
    changes = df.loc[(np.sign(df["Signal"]).diff().ne(0))]
    changes["Change"] = 1
    return df.reset_index().merge(changes, how="left")


def get_last_signal(df: pd.DataFrame) -> int:
    last = df.iloc[-1, :]
    if last["Change"] == 1.0:
        return int(last["Signal"])
    return 0


def backtesting(df: pd.DataFrame) -> Any:
    df.set_index('Date', inplace=True)
    df['Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['Strategy'] = df['Signal'].shift(1) * df['Returns']
    df['pct_returns'] = df['Adj Close'].pct_change()
    df['pct_strategy'] = df['Signal'].shift(1) * df['pct_returns']
    # cumulative_strategy_returns = (df['pct_strategy'] + 1).cumprod()
    # cumulative_strategy_returns.plot(figsize=(10, 7))
    # cumulative_baseline_returns = (df['Returns'] + 1).cumprod()
    # cumulative_baseline_returns.plot(figsize=(10, 7))
    # plt.show()

    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['returns'] * df['Signal'].shift(1)

    df_strat = df.__deepcopy__()
    df_strat.dropna(inplace=True)
    perf = np.exp(df_strat[['Returns', 'Strategy']].sum())
    out = perf['Strategy'] - perf['Returns']
    # print(f'Baseline Performance: {perf["Returns"]}\nStrategy Performance: {perf["Strategy"]}\nDifference: {out}')

    return calculate_bt_metrics(df, perf, out)


def backtest_short_long(df: pd.DataFrame, short: int, long: int):
    """https://blog.quantinsti.com/backtesting/"""
    df['short_mavg'] = df['Close'].rolling(short).mean()
    df['long_mavg'] = df['Close'].rolling(long).mean()
    df['long_positions'] = np.where(df['short_mavg'] > df['long_mavg'], 1, 0)
    df['short_positions'] = np.where(df['short_mavg'] < df['long_mavg'], -1, 0)
    df['positions'] = df['long_positions'] + df['short_positions']

    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['returns'] * df['positions'].shift(1)

    plot_cum_returns(df)

    return calculate_bt_metrics(df)


def plot_cum_returns(df: pd.DataFrame) -> None:
    plot_data = df[-3000:]
    plt.figure(figsize=(10, 7))
    plt.title('Long and Short Signal', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(plot_data['Close'], label='Close')
    plt.plot(plot_data['short_mavg'], label='50-Day Moving Average')
    plt.plot(plot_data['long_mavg'], label='200-day Moving Average')
    plt.plot(plot_data[(plot_data['long_positions'] == 1) &
                       (plot_data['long_positions'].shift(1) == 0)]['short_mavg'],
             '^', ms=15, label='Buy Signal', color='green')
    plt.plot(plot_data[(plot_data['short_positions'] == -1) &
                       (plot_data['short_positions'].shift(1) == 0)]['short_mavg'],
             '^', ms=15, label='Sell Signal', color='red')
    plt.legend()
    plt.show()


def calculate_bt_metrics(df: pd.DataFrame, perf: List[float], out: float) -> Tuple[Any]:
    cumulative_returns = (df['strategy_returns'] + 1).cumprod()
    days = len(cumulative_returns)
    annualised_returns = (cumulative_returns.iloc[-1] ** (252 / days) - 1) * 100
    annualised_volatility = np.std(df.strategy_returns) * (252 ** 0.5) * 100
    risk_free_rate = 0.01 / 252
    sharpe_ratio = np.sqrt(252) * (np.mean(df.strategy_returns) - (risk_free_rate)) / np.std(df.strategy_returns)
    running_max = np.maximum.accumulate(cumulative_returns.dropna())
    running_max[running_max < 1] = 1
    drawdown = (cumulative_returns) / running_max - 1
    max_dd = drawdown.min() * 100
    return (annualised_returns, annualised_volatility, sharpe_ratio, max_dd, perf[0], perf[1], out)


def find_sma_combo(df0: pd.DataFrame):
    sma1 = range(5, 80, 1)
    sma2 = range(30, 281, 1)

    results = pd.DataFrame()
    for SMA1, SMA2 in product(sma1, sma2):
        df = df0.__deepcopy__()
        df['Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        df['SMA1'] = df['Adj Close'].rolling(SMA1).mean()
        df['SMA2'] = df['Adj Close'].rolling(SMA2).mean()
        df.dropna(inplace=True)
        df['Position'] = np.where(df['SMA1'] > df['SMA2'], 1, -1)
        df['Strategy'] = df['Position'].shift(1) * df['Returns']
        df.dropna(inplace=True)
        perf = np.exp(df[['Returns', 'Strategy']].sum())
        results = results._append(pd.DataFrame(
            {'SMA1': SMA1, 'SMA2': SMA2, 'Market': perf['Returns'], 'Strategy': perf['Strategy'],
             'Out': perf['Strategy'] - perf['Returns']},
            index=[0]), ignore_index=True)
    results.sort_values('Out', ascending=False).head(10)
    return results


def plot_all_strategies(outputs: Dict[str, Any]) -> None:
    nr = round((len(outputs) // 2), 0)
    fig, axs = plt.subplots(nrows=nr, ncols=2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Backtesting strategies against currency pairs", fontsize=18, y=0.95)

    for name, ticker, ax in zip(outputs.keys(), outputs.values(), axs.ravel()):
        ticker = ticker.set_index('Date')
        ticker["Adj Close"].plot(ax=ax, label="Currency Price", alpha=0.5)
        ticker["Signal"].plot(ax=ax, label="Signal", style='o', secondary_y=True)

        # chart formatting
        ax.set_title(name)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

    plt.show()


def get_candles(instrument: Pair, n: int) -> List[Candle]:
    client = CandleClient(access_token, real=False)
    collector = client.get_collector(instrument, Gran.D)
    return collector.grab(n)


def candle_to_df(candles):
    dfstream = pd.DataFrame(columns=["Open", "Close", "High", "Low"])

    i = 0
    for candle in candles:
        dfstream.loc[i, ["Open"]] = float(str(candle.bid.o))
        dfstream.loc[i, ["Close"]] = float(str(candle.bid.c))
        dfstream.loc[i, ["High"]] = float(str(candle.bid.h))
        dfstream.loc[i, ["Low"]] = float(str(candle.bid.l))
        i += 1

    dfstream["Open"] = dfstream["Open"].astype(float)
    dfstream["Close"] = dfstream["Close"].astype(float)
    dfstream["High"] = dfstream["High"].astype(float)
    dfstream["Low"] = dfstream["Low"].astype(float)
    dfstream["Adj Close"] = dfstream["Close"].astype(float)
    return dfstream


def get_limits(dfstream: pd.DataFrame, candle: Candle) -> List[float]:
    SLTPratio = 5.0
    previous_candleR = abs(dfstream["High"].iloc[-2] - dfstream["Low"].iloc[-2])

    SLBuy = float(str(candle.bid.o)) - previous_candleR
    SLSell = float(str(candle.bid.o)) + previous_candleR

    TPBuy = float(str(candle.bid.o)) + previous_candleR * SLTPratio
    TPSell = float(str(candle.bid.o)) - previous_candleR * SLTPratio

    return [SLBuy, SLSell, TPBuy, TPSell]


def trade(client: API, instrument: str, units: int, tp: int, sl: int):
    mo = MarketOrderRequest(
        instrument=instrument,
        units=units,
        takeProfitOnFill=TakeProfitDetails(price=tp).data,
        # stopLossOnFill=StopLossDetails(price=sl).data,
    )
    r = orders.OrderCreate(account_id, data=mo.data)
    rv = client.request(r)
    print(rv)


def trading_job(instrument: Tuple[str, Pair], strategy: Callable, active: bool = False):
    candles = get_candles(instrument[1], 110)
    dfstream = candle_to_df(candles)
    signals = get_signals(dfstream.iloc[:-1, :], strategy)
    signal = get_last_signal(signals)

    candle = candles[-1]
    client = API(access_token)
    SLBuy, SLSell, TPBuy, TPSell = get_limits(dfstream, candle)
    if int(signal) == 0:
        print(f"No triggers for {instrument[0]}")
    elif int(signal) == -1:
        print(f"Selling {instrument[0]}")
        if active:
            trade(client, instrument[0], -1000, TPSell, SLSell)
    elif int(signal) == 1:
        print(f"Buying {instrument[0]}")
        if active:
            trade(client, instrument[0], 1000, TPBuy, SLBuy)


def backtest_all(strategies: List[Callable], currency_pairs: Dict[str, Pair]):
    res_df = pd.DataFrame()
    for strategy in strategies:
        for pair in currency_pairs.items():
            df = get_data(pair[0])
            signals = get_signals(df, strategy)
            # a = backtest_short_long(df)
            a = backtesting(signals)
            res_df = res_df._append(pd.DataFrame(
                {
                    "Strategy": strategy.__name__,
                    "Currency": str(pair[0]),
                    "Annualised Return": a[0],
                    "Annualised Volatility": a[1],
                    "Sharpe Ratio": a[2],
                    "Max Drawdown": a[3],
                    "Baseline Performance": a[4],
                    "Strategy Performance": a[5],
                    "Difference": a[6]
                }, index=[0]), ignore_index=True)
    res_df.head()
    grouped = res_df.groupby(['Strategy']).mean(["Annualised Return",
                                                 "Annualised Volatility",
                                                 "Sharpe Ratio",
                                                 "Max Drawdown",
                                                 "Baseline Performance",
                                                 "Strategy Performance",
                                                 "Difference"])
    print(grouped.head())



if __name__ == "__main__":
    currency_pairs = {
        "GBPUSD=X": ("GBP_USD", Pair.GBP_USD),
        "EURGBP=X": ("EUR_GBP", Pair.EUR_GBP),
        "EURUSD=X": ("EUR_USD", Pair.EUR_USD),
        "JPY=X": ("USD_JPY", Pair.USD_JPY),
        "AUDUSD=X": ("AUD_USD", Pair.AUD_USD),
        "NZDUSD=X": ("NZD_USD", Pair.NZD_USD),
        "EURJPY=X": ("EUR_JPY", Pair.EUR_JPY),
        "EURCAD=X": ("EUR_CAD", Pair.EUR_CAD),
        "EURCHF=X": ("EUR_CHF", Pair.EUR_CHF),  # Swiss Frank
    }
    strategies = [strategy_sma_9_61, strategy_red_white_blue]  #
    outputs = {}
    # backtest_all(strategies, currency_pairs)
    for strategy, pair in product(strategies, currency_pairs.items()):
        print(f'{"*" * 30}{strategy.__name__}: {pair[1][0]}{"*" * 30}')
        df = get_data(pair[0])
        # find_sma_combo(df)  # output was 9d, 61d
        signals = get_signals(df, strategy)
        outputs[f"{strategy.__name__}: {pair[1][0]}"] = signals
        trading_job(pair[1], strategy, False)
    # plot_all_strategies(outputs)
