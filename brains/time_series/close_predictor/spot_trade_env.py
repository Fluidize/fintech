import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
from rich import print

import plotly.graph_objects as go

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
        self.pct_pnl_history: List[float] = []

    def buy(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> bool:
        cost = quantity * price
        if cost > self.cash:
            print(f"[red]Insufficient funds for buy order. Required: ${cost:.2f}, Available: ${self.cash:.2f}[/red]")
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
        return True
    
    def sell(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> bool:
        if symbol not in self.positions:
            print(f"[red]No position found for {symbol}[/red]")
            return False
            
        pos = self.positions[symbol]
        if quantity > pos.quantity:
            print(f"[red]Insufficient shares for sell order. Required: {quantity}, Available: {pos.quantity}[/red]")
            return False
            
        proceeds = quantity * price
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

    def buy_max(self, symbol: str, price: float, timestamp: datetime) -> bool:
        if self.cash <= 0:
            return False
        
        quantity = self.cash / price
        return self.buy(symbol, quantity, price, timestamp)

    def sell_max(self, symbol: str, price: float, timestamp: datetime) -> bool:
        if symbol not in self.positions:
            return False
        
        quantity = self.positions[symbol].quantity
        return self.sell(symbol, quantity, price, timestamp)

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
                start_date = (datetime.now() - timedelta(days=8) - timedelta(days=8*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
                end_date = (datetime.now() - timedelta(days=8*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
                temp_data = yf.download(symbol, start=start_date, end=end_date, interval=self.interval)
                data = pd.concat([data, temp_data])
            
            data.sort_index(inplace=True)
            data.columns = data.columns.droplevel(1)
            data.reset_index(inplace=True)
            data.rename(columns={'index': 'Datetime'}, inplace=True)
            self.data[symbol] = pd.DataFrame(data)
            
            self.context[symbol] = self.data[symbol].iloc[:self.context_length].copy()
            
        # Find the minimum length among all datasets
        min_length = min(len(df) for df in self.data.values())
        
        # Trim all datasets to the same length
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
        if self.current_index >= len(self.data[self.symbols[0]]) - 1:
            return False
            
        self.current_index += 1
        
        # Update context for each symbol
        for symbol in self.symbols:
            current_data = self.data[symbol].iloc[self.current_index]
            self.context[symbol] = pd.concat([
                self.context[symbol].iloc[1:],  # Remove oldest row
                pd.DataFrame([current_data])    # Add new row
            ]).reset_index(drop=True)
        
        current_prices = self.get_current_prices()
        self.portfolio.update_positions(current_prices)
        self.portfolio.pct_pnl_history.append(self.portfolio.total_profit_loss_pct)
        return True
    
    def reset(self):
        self.current_index = 0
        self.portfolio = Portfolio(self.portfolio.initial_capital)
        # Reset context for each symbol
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

    def create_performance_plot(self, show_graph: bool = False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data[self.symbols[0]]['Datetime'], y=self.portfolio.pct_pnl_history, mode='lines', name='Strategy PnL'))
        
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
        
        fig.update_layout(title='Portfolio Performance', xaxis_title='Time', yaxis_title='Profit/Loss (%)')
        if show_graph:
            fig.show()
        return fig

def MA_Crossover_Strategy(env: TradingEnvironment, context, prices):
    ma10 = context['Close'].rolling(window=10).mean().iloc[-1]
    ma20 = context['Close'].rolling(window=20).mean().iloc[-1]
    current_close = prices['Close']
    
    if ma10 > ma20:
        env.portfolio.buy_max('BTC-USD', current_close)
    else:
        env.portfolio.sell_max('BTC-USD', current_close)

def RSI_Strategy(env: TradingEnvironment, context, prices):
    current_close = prices['Close']

    delta_p = context['Close'].diff()
    gain = delta_p.where(delta_p > 0, 0)
    loss = -delta_p.where(delta_p < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)

    current_rsi = rsi.iloc[-1]

    if current_rsi < 30:
        env.portfolio.buy('BTC-USD',0.1, current_close, env.get_current_timestamp())
    elif current_rsi > 70:
        env.portfolio.sell_max('BTC-USD', current_close, env.get_current_timestamp())

def Custom_Strategy(env: TradingEnvironment, context, prices):
    current_close = prices['Close']

    if context['Close'].iloc[-1] > context['Close'].iloc[-2]:
        env.portfolio.buy_max('BTC-USD', current_close, env.get_current_timestamp())

if __name__ == "__main__":
    # from close_only import ModelTesting
    # model = ModelTesting(ticker=None, chunks=None, interval=None, age_days=None)
    # model.load_model('BTC-USD_5m_11476066.keras')

    env = TradingEnvironment(
        symbols=['BTC-USD'],
        initial_capital=100000.0,
        chunks=3,
        interval='5m',
        age_days=1
    )
    env.fetch_data() #init data
            
    while env.step():
        current_state = env.get_state()
        context = current_state['context']['BTC-USD'] #context includes current price
        prices = current_state['prices']['BTC-USD'] #current priceA

        RSI_Strategy(env, context, prices)
        print(len(env.portfolio.trade_history))
        
    env.create_performance_plot(show_graph=True)