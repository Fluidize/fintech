import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
from rich import print
from tqdm import tqdm
import plotly.graph_objects as go

from time_series.close_predictor.close_only import ModelTesting

@dataclass
class Position:
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    
    @property
    def value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def profit_loss(self) -> float:
        return self.quantity * (self.current_price - self.average_price)
    
    @property
    def profit_loss_pct(self) -> float:
        return ((self.current_price - self.average_price) / self.average_price) * 100

class Portfolio:
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[float] = []
        self.pnl_history: List[float] = []
        self.pct_pnl_history: List[float] = []

    def buy(self, symbol: str, quantity: float, price: float, timestamp: datetime, verbose: bool = False) -> bool:
        cost = round(quantity * price, 4) #roundoff buy errors
        if cost > self.cash:
            if verbose:
                print(f"[red]Insufficient funds for buy order. Required: ${cost}, Available: ${self.cash}[/red]")
            return False
            
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_quantity = pos.quantity + quantity
            total_cost = (pos.quantity * pos.average_price) + (quantity * price)
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                average_price=total_cost / total_quantity,
                current_price=price
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                current_price=price
            )
            
        self.cash -= cost
        self.trade_history.append({
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'cash_remaining': self.cash,
            'timestamp': timestamp
        })
        if verbose:
            print(f"[green]Bought {quantity} shares of {symbol} at {price:.2f} for {cost:.2f}[/green]")
        return True
    
    def sell(self, symbol: str, quantity: float, price: float, timestamp: datetime, verbose: bool = False) -> bool:
        if symbol not in self.positions:
            if verbose:
                print(f"[red]No position found for {symbol}[/red]")
            return False
            
        pos = self.positions[symbol]
        if quantity > pos.quantity:
            if verbose:
                print(f"[red]Insufficient shares for sell order. Required: {quantity}, Available: {pos.quantity}[/red]")
            return False
            
        proceeds = round(quantity * price, 4)
        self.cash += proceeds
        
        if quantity == pos.quantity:
            del self.positions[symbol]
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=pos.quantity - quantity,
                average_price=pos.average_price,
                current_price=price
            )
            
        self.trade_history.append({
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': price,
            'proceeds': proceeds,
            'cash_remaining': self.cash,
            'timestamp': timestamp
        })
        if verbose:
            print(f"[red]Sold {quantity} shares of {symbol} at {price:.2f} for {proceeds:.2f}[/red]")
        return True
    
    def update_positions(self, current_prices: Dict[str, pd.Series]):
        for symbol, price_data in current_prices.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                # Extract Close price from the OHLCV data
                current_price = float(price_data['Close'])
                self.positions[symbol] = Position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    average_price=pos.average_price,
                    current_price=current_price
                )
    
    @property
    def total_value(self) -> float:
        position_value = sum(pos.value for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def total_profit_loss(self) -> float:
        return self.total_value - self.initial_capital
    
    @property
    def total_profit_loss_pct(self) -> float:
        return (self.total_profit_loss / self.initial_capital) * 100
    
    def get_position_summary(self) -> str:
        if not self.positions:
            return "No open positions"
            
        summary = "Current Positions:\n"
        for symbol, pos in self.positions.items():
            summary += f"{symbol}: {pos.quantity:.4f} shares @ ${pos.average_price:.2f} "
            summary += f"(Current: ${pos.current_price:.2f}, P/L: ${pos.profit_loss:.2f} [{pos.profit_loss_pct:.2f}%])\n"
        return summary

    def buy_max(self, symbol: str, price: float, timestamp: datetime, verbose: bool = False) -> bool:
        if self.cash <= 0:
            return False
        
        quantity = self.cash / price
        return self.buy(symbol, quantity, price, timestamp, verbose)

    def sell_max(self, symbol: str, price: float, timestamp: datetime, verbose: bool = False) -> bool:
        if symbol not in self.positions:
            return False
        
        quantity = self.positions[symbol].quantity
        return self.sell(symbol, quantity, price, timestamp, verbose)

class TradingEnvironment:
    def __init__(self, symbols: List[str], initial_capital: float = 10000.0, chunks: int = 5, interval: str = '5m', age_days: int = 10):
        self.symbols = symbols
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.portfolio = Portfolio(initial_capital)
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_index = 0
        self.context_length = 100
        self.context: Dict[str, pd.DataFrame] = {}  # Store context for each symbol
        
    def fetch_data(self):
        for symbol in self.symbols:
            data = pd.DataFrame()
            for x in range(self.chunks):
                chunksize = 1
                start_date = (datetime.now() - timedelta(days=chunksize) - timedelta(days=chunksize*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
                end_date = (datetime.now() - timedelta(days=chunksize*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
                temp_data = yf.download(symbol, start=start_date, end=end_date, interval=self.interval)
                data = pd.concat([data, temp_data])
            
            data.sort_index(inplace=True)
            data.columns = data.columns.droplevel(1)
            data.reset_index(inplace=True)
            data.rename(columns={'index': 'Datetime'}, inplace=True)
            data.rename(columns={'Date': 'Datetime'}, inplace=True)
            self.data[symbol] = pd.DataFrame(data)  
            self.context[symbol] = self.data[symbol].iloc[:self.context_length].copy()

        min_length = min(len(df) for df in self.data.values())
        for symbol in self.symbols:
            self.data[symbol] = self.data[symbol].iloc[-min_length:]
            
        print(f"Fetched {min_length} data points for each symbol")

    def fetch_1d_data(self, days: int = 365):
        for symbol in self.symbols:
            data = yf.download(symbol, start=datetime.now() - timedelta(days=days), end=datetime.now(), interval='1d')
            data.sort_index(inplace=True)
            data.columns = data.columns.droplevel(1)
            data.reset_index(inplace=True)
            data.rename(columns={'index': 'Datetime'}, inplace=True)
            data.rename(columns={'Date': 'Datetime'}, inplace=True) #1d data has Date instead of Datetime
            self.data[symbol] = pd.DataFrame(data)
            self.context[symbol] = self.data[symbol].iloc[:self.context_length].copy()
        
        min_length = min(len(df) for df in self.data.values())
        for symbol in self.symbols:
            self.data[symbol] = self.data[symbol].iloc[-min_length:]
            
        print(f"Fetched {min_length} data points for each symbol")
    
    def get_current_prices(self) -> Dict[str, pd.Series]:
        return {
            symbol: self.data[symbol].iloc[self.current_index]
            for symbol in self.symbols
        }
    
    def get_current_timestamp(self) -> datetime:
        return self.data[self.symbols[0]].iloc[self.current_index]['Datetime']
    
    def step(self) -> bool:
        extended_context = False

        if self.current_index >= len(self.data[self.symbols[0]]) - 1:
            return False
            
        self.current_index += 1
        
        if extended_context:
            for symbol in self.symbols:
                current_data = self.data[symbol].iloc[self.current_index]
                self.context[symbol] = pd.concat([
                    self.context[symbol],
                    pd.DataFrame([current_data])
                ]).reset_index(drop=True)
        else:
            for symbol in self.symbols:
                current_data = self.data[symbol].iloc[self.current_index]
                self.context[symbol] = pd.concat([
                    self.context[symbol][-self.context_length+1:],
                    pd.DataFrame([current_data])
                ]).reset_index(drop=True)
        
        current_prices = self.get_current_prices()
        self.portfolio.update_positions(current_prices)
        self.portfolio.pnl_history.append(self.portfolio.total_profit_loss)
        self.portfolio.pct_pnl_history.append(self.portfolio.total_profit_loss_pct)
        return True
    
    def reset(self):
        self.current_index = 0
        self.portfolio = Portfolio(self.portfolio.initial_capital)
        for symbol in self.symbols:
            self.context[symbol] = self.data[symbol].iloc[:self.context_length].copy()
    
    def get_state(self) -> Dict:
        return {
            'timestamp': self.get_current_timestamp(),
            'prices': self.get_current_prices(),
            'context': self.context,  # Add context to state
            'portfolio_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions': self.portfolio.positions,
            'trade_history': self.portfolio.trade_history
        }

    def get_summary(self) -> str:
        max_drawdown = max(self.portfolio.pct_pnl_history) - min(self.portfolio.pct_pnl_history)
        gains = list(filter(lambda x: x > 0, self.portfolio.pnl_history))
        losses = list(filter(lambda x: x < 0, self.portfolio.pnl_history))

        if len(gains) == 0 or len(losses) == 0:
            profit_factor = np.nan
            RR_ratio = np.nan
            win_rate = np.nan
            optimal_wr = np.nan
        else:
            profit_factor = sum(gains) / abs(sum(losses))
            RR_ratio = (sum(gains) / len(gains)) / abs(sum(losses) / len(losses))
            win_rate = len(gains) / (len(gains) + len(losses))
            optimal_wr = 1 / (RR_ratio + 1)
        
        return f"{self.symbols[0]} {self.interval} | PnL: {self.portfolio.total_profit_loss_pct:.2f}% | Max DD: {max_drawdown:.2f}% | PF: {profit_factor:.2f} | RR Ratio:{RR_ratio:.2f} | WR: {win_rate:.2f} | Optimal WR: {optimal_wr:.2f}"

    def create_performance_plot(self, show_graph: bool = False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data[self.symbols[0]]['Datetime'], y=self.portfolio.pct_pnl_history, mode='lines', name='Strategy %'))
        
        #price line %
        start_price = self.data[self.symbols[0]].iloc[0]['Close']
        start_pct_change = ((self.data[self.symbols[0]]['Close'] - start_price) / start_price) * 100
        fig.add_trace(go.Scatter(x=self.data[self.symbols[0]]['Datetime'], y=start_pct_change, mode='lines', name='Price %', line=dict(color='orange')))
        
        # Add buy/sell markers
        buy_x = []
        buy_y = []
        sell_x = []
        sell_y = []
        
        for trade in self.portfolio.trade_history:
            # Find the index in the data that matches the trade timestamp
            idx = self.data[self.symbols[0]]['Datetime'].searchsorted(trade['timestamp'])
            if trade['action'] == 'BUY':
                buy_x.append(trade['timestamp'])
                buy_y.append(start_pct_change[idx])
            else:  # SELL
                sell_x.append(trade['timestamp'])
                sell_y.append(start_pct_change[idx])
        
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='Buy', 
                               marker=dict(color='green', size=5, symbol='circle')))
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='Sell', 
                               marker=dict(color='red', size=5, symbol='circle')))

        fig.update_layout(title=f"Portfolio Performance | {self.get_summary()}", xaxis_title='Time', yaxis_title='Profit/Loss (%)')
        if show_graph:
            fig.show()
        return fig

class Backtest:
    def __init__(self, env: TradingEnvironment):
        self.env = env

    def _calculate_rsi(self, context, period=14):
        delta_p = context['Close'].diff()
        gain = delta_p.where(delta_p > 0, 0)
        loss = -delta_p.where(delta_p < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        return rsi
    
    def _calculate_macd(self, context, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = context['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = context['Close'].ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram
        
    def MA(self, env: TradingEnvironment, context, current_ohlcv):
        ma50 = context['Close'].rolling(window=50).mean().iloc[-1]
        ma100 = context['Close'].rolling(window=200).mean().iloc[-1]

        current_close = current_ohlcv['Close']

        if current_close > ma50 and current_close > ma100:
            env.portfolio.buy_max('BTC-USD', current_close, env.get_current_timestamp())
        elif current_close < ma50 and current_close < ma100:
            env.portfolio.sell_max('BTC-USD', current_close, env.get_current_timestamp())

    def RSI(self, env: TradingEnvironment, context, current_ohlcv):
        current_close = current_ohlcv['Close']

        rsi = self._calculate_rsi(context, 28)
        current_rsi = rsi.iloc[-1]

        if current_rsi < 30:
            env.portfolio.buy('BTC-USD',0.1, current_close, env.get_current_timestamp())
        elif current_rsi > 70:
            env.portfolio.sell_max('BTC-USD', current_close, env.get_current_timestamp())


    def MACD(self, env: TradingEnvironment, context, current_ohlcv):
        current_close = current_ohlcv['Close']

        macd_line, signal_line, histogram = self._calculate_macd(context)
        current_macd_line = macd_line.iloc[-1]
        current_signal_line = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        if current_macd_line > current_signal_line and current_histogram > 30:
            env.portfolio.buy_max('BTC-USD', current_close, env.get_current_timestamp())
        elif current_macd_line < current_signal_line and current_histogram < -30:
            env.portfolio.sell_max('BTC-USD', current_close, env.get_current_timestamp())
        
    def Custom(self, env: TradingEnvironment, context, current_ohlcv):
        current_close = current_ohlcv['Close']

        rsi = self._calculate_rsi(context, 28)
        current_rsi = rsi.iloc[-1]
        macd_line, signal_line, histogram = self._calculate_macd(context)
        current_macd_line = macd_line.iloc[-1]
        current_signal_line = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]

        buy_conditions = [current_close > context['Close'].iloc[-2], current_macd_line > current_signal_line]
        sell_conditions = [current_close < context['Close'].iloc[-2], current_macd_line < current_signal_line]

        if all(buy_conditions):
            env.portfolio.buy_max('BTC-USD', current_close, env.get_current_timestamp())
        elif all(sell_conditions):
            env.portfolio.sell_max('BTC-USD', current_close, env.get_current_timestamp())

    def RNN(self, env: TradingEnvironment, context, current_ohlcv, model):
        input_data, prediction = model.run(context)
        current_close = current_ohlcv['Close']

        if prediction > current_close:
            env.portfolio.buy_max('BTC-USD', current_close, env.get_current_timestamp())
        else:
            env.portfolio.sell_max('BTC-USD', current_close, env.get_current_timestamp())
    
    def Perfect(self, env, context, current_ohlcv):
        #perfect strategy
        index = env.current_index
        try:
            if context['Close'].iloc[-1] < env.data[env.symbols[0]]['Close'].iloc[index+1]:
                env.portfolio.buy_max('BTC-USD', current_ohlcv['Close'], env.get_current_timestamp())
            elif context['Close'].iloc[-1] > env.data[env.symbols[0]]['Close'].iloc[index+1]:
                env.portfolio.sell_max('BTC-USD', current_ohlcv['Close'], env.get_current_timestamp())
            print(env.portfolio.total_profit_loss_pct)
        except:
            pass
    

    def run(self, strategy=None):
        env = self.env

        env.fetch_data()
        print("Starting Backtest")
        progress_bar = tqdm(total=len(env.data[env.symbols[0]]))

        # model = ModelTesting(ticker=None, chunks=None, interval=None, age_days=None)
        # model.load_model(r'BTC-USD_5m_11476066.keras')

        while env.step():
            current_state = env.get_state()
            context = current_state['context'][env.symbols[0]] #context includes current price
            current_ohlcv = current_state['prices'][env.symbols[0]] #current ohlcv

            self.MACD(env, context, current_ohlcv)
            progress_bar.update(1)
        progress_bar.close()

        print(env.get_summary())

        env.create_performance_plot(show_graph=True)

env = TradingEnvironment(symbols=['BTC-USD'], initial_capital=100000, chunks=29, interval='5m', age_days=10)
backtest = Backtest(env)
backtest.run()
