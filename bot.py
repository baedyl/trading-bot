import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import talib
import streamlit as st
import time
import threading
from datetime import datetime, timedelta

class AdvancedTradingStrategy:
    def __init__(self, ticker='AAPL', start_date='2021-01-01', end_date=None):
        self.ticker = ticker
        self.start_date = pd.to_datetime(start_date)
        
        # Ensure end date is not in the future
        if end_date is None:
            self.end_date = pd.Timestamp.today()
        else:
            proposed_end_date = pd.to_datetime(end_date)
            self.end_date = min(proposed_end_date, pd.Timestamp.today() + pd.DateOffset(years=1))
        
        print(f"Initializing Trading Strategy for {ticker}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        
        self.data = None
        self.full_data = None  # Store full data for indicator calculation
        self.signals = None
        self.indicators = None
        self.support_resistance = None
    
    def fetch_historical_data(self):
        try:
            print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
            
            # Fetch historical data with extended start date for indicators
            extended_start_date = self.start_date - pd.DateOffset(days=400)  # Enough for 200-day MA
            
            # For future-looking scenarios, use predicted/simulated data
            if self.end_date > pd.Timestamp.today():
                print("Future date detected, fetching historical and simulating future data")
                # Fetch historical data and apply a simple predictive model
                historical_data = yf.download(
                    self.ticker, 
                    start=extended_start_date, 
                    end=pd.Timestamp.today(),
                    progress=False
                )
                
                if historical_data.empty:
                    print("No historical data received from yfinance")
                    return None
                
                # Simple trend projection
                last_price = historical_data['Close'].iloc[-1]
                historical_returns = historical_data['Close'].pct_change()
                avg_return = historical_returns.mean()
                std_return = historical_returns.std()
                
                # Generate simulated future prices
                future_dates = pd.date_range(
                    start=pd.Timestamp.today() + pd.DateOffset(days=1), 
                    end=self.end_date, 
                    freq='B'
                )
                
                future_prices = [last_price]
                for _ in range(len(future_dates)):
                    # Simulate future price with random walk
                    price_change = np.random.normal(avg_return, std_return)
                    next_price = future_prices[-1] * (1 + price_change)
                    future_prices.append(next_price)
                
                # Create DataFrame for future prices
                future_data = pd.DataFrame({
                    'Open': future_prices[1:],
                    'High': [p * 1.01 for p in future_prices[1:]],
                    'Low': [p * 0.99 for p in future_prices[1:]],
                    'Close': future_prices[1:],
                    'Volume': [historical_data['Volume'].mean()] * len(future_dates)
                }, index=future_dates)
                
                # Combine historical and projected data
                self.full_data = pd.concat([historical_data, future_data])
            else:
                print("Fetching normal historical data")
                # Normal historical data download
                self.full_data = yf.download(
                    self.ticker, 
                    start=extended_start_date, 
                    end=self.end_date,
                    progress=False
                )
                
                if self.full_data.empty:
                    print("No data received from yfinance")
                    return None
            
            # Handle multi-level columns if present
            if isinstance(self.full_data.columns, pd.MultiIndex):
                self.full_data.columns = self.full_data.columns.droplevel(1)
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in self.full_data.columns:
                    print(f"Missing required column: {col}")
                    return None
            
            # Store trimmed data for display
            self.data = self.full_data.loc[self.start_date:self.end_date].copy()
            
            if self.data.empty:
                print("No data available for the selected date range")
                return None
            
            print("Full Data Shape (for indicators):", self.full_data.shape)
            print("Display Data Shape:", self.data.shape)
            print("Date Range:", self.data.index[0], "to", self.data.index[-1])
            
            return self.data
        except Exception as e:
            print(f"Data fetching error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_support_resistance(self):
        """Calculate dynamic support and resistance levels"""
        if self.full_data is None or self.full_data.empty:
            print("No data available for support/resistance calculation")
            return None
        
        # Calculate support and resistance using rolling highs/lows
        support_resistance_full = pd.DataFrame(index=self.full_data.index)
        support_resistance_full['support'] = self.full_data['Low'].rolling(window=20).min()
        support_resistance_full['resistance'] = self.full_data['High'].rolling(window=20).max()
        
        # Trim to display period
        self.support_resistance = support_resistance_full.loc[self.start_date:self.end_date]
        
        return self.support_resistance
    
    def calculate_indicators(self):
        if self.full_data is None or self.full_data.empty:
            print("No data available for indicator calculation")
            return {}
        
        indicators_full = {}
        
        # Calculate indicators on full data
        close = self.full_data['Close'].values
        high = self.full_data['High'].values
        low = self.full_data['Low'].values
        
        # Moving Averages
        indicators_full['MA_20'] = talib.SMA(close, timeperiod=20)
        indicators_full['MA_50'] = talib.SMA(close, timeperiod=50)
        indicators_full['MA_200'] = talib.SMA(close, timeperiod=200)
        
        # RSI
        indicators_full['RSI'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators_full['MACD'] = macd
        indicators_full['MACD_Signal'] = macdsignal
        indicators_full['MACD_Hist'] = macdhist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        indicators_full['BB_Upper'] = upper
        indicators_full['BB_Middle'] = middle
        indicators_full['BB_Lower'] = lower
        
        # Trim indicators to display period
        self.indicators = {}
        for key, values in indicators_full.items():
            series = pd.Series(values, index=self.full_data.index)
            self.indicators[key] = series.loc[self.start_date:self.end_date]
        
        # Debug info
        print("\nIndicator Statistics:")
        for key, series in self.indicators.items():
            valid_count = series.notna().sum()
            print(f"{key}: {valid_count} valid values out of {len(series)}")
        
        return self.indicators
    
    def generate_signals(self):
        if not self.indicators:
            print("No indicators available for signal generation")
            return pd.DataFrame()
        
        # Calculate support/resistance if not already done
        if not hasattr(self, 'support_resistance') or self.support_resistance is None:
            self.calculate_support_resistance()
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0
        signals['confirmed_signal'] = 0
        
        # Get moving averages
        ma_20 = self.indicators['MA_20']
        ma_50 = self.indicators['MA_50']
        
        # Forward fill NaN values
        ma_20 = ma_20.ffill()
        ma_50 = ma_50.ffill()
        
        # Define crossover functions
        def crossover_up(series1, series2):
            curr = series1 > series2
            prev = series1.shift(1) <= series2.shift(1)
            return curr & prev
        
        def crossover_down(series1, series2):
            curr = series1 < series2
            prev = series1.shift(1) >= series2.shift(1)
            return curr & prev
        
        # Generate initial signals on moving average crossovers
        buy_conditions = crossover_up(ma_20, ma_50)   # 20MA crosses above 50MA
        sell_conditions = crossover_down(ma_20, ma_50) # 20MA crosses below 50MA
        
        # Set initial signals
        signals.loc[buy_conditions, 'signal'] = 1
        signals.loc[sell_conditions, 'signal'] = -1
        
        # Enhanced signal confirmation with support/resistance validation
        for i in range(1, len(signals)):
            if signals['signal'].iloc[i] == 1:  # Buy signal
                # Confirm if price breaks above resistance
                if (self.data['Close'].iloc[i] > self.support_resistance['resistance'].iloc[i-1] and
                    self.data['Close'].iloc[i-1] <= self.support_resistance['resistance'].iloc[i-1]):
                    signals['confirmed_signal'].iloc[i] = 1
            elif signals['signal'].iloc[i] == -1:  # Sell signal
                # Confirm if price breaks below support
                if (self.data['Close'].iloc[i] < self.support_resistance['support'].iloc[i-1] and
                    self.data['Close'].iloc[i-1] >= self.support_resistance['support'].iloc[i-1]):
                    signals['confirmed_signal'].iloc[i] = -1
        
        # Store signals
        self.signals = signals
        
        # Print Signal Details
        buy_signal_dates = signals[signals['signal'] == 1].index
        sell_signal_dates = signals[signals['signal'] == -1].index
        confirmed_buy_dates = signals[signals['confirmed_signal'] == 1].index
        confirmed_sell_dates = signals[signals['confirmed_signal'] == -1].index
        
        print("\nMoving Average Crossover Signals:")
        print(f"Total Buy Signals (20MA crosses above 50MA): {len(buy_signal_dates)}")
        if len(buy_signal_dates) > 0:
            print("Buy Signal Dates:")
            for date in buy_signal_dates:
                price = self.data.loc[date, 'Close']
                ma20_val = ma_20.loc[date]
                ma50_val = ma_50.loc[date]
                print(f"  {date.date()}: ${price:.2f} (MA20: ${ma20_val:.2f}, MA50: ${ma50_val:.2f})")
        
        print(f"\nTotal Sell Signals (20MA crosses below 50MA): {len(sell_signal_dates)}")
        if len(sell_signal_dates) > 0:
            print("Sell Signal Dates:")
            for date in sell_signal_dates:
                price = self.data.loc[date, 'Close']
                ma20_val = ma_20.loc[date]
                ma50_val = ma_50.loc[date]
                print(f"  {date.date()}: ${price:.2f} (MA20: ${ma20_val:.2f}, MA50: ${ma50_val:.2f})")
        
        print(f"\nConfirmed Buy Signals (with S/R validation): {len(confirmed_buy_dates)}")
        print(f"Confirmed Sell Signals (with S/R validation): {len(confirmed_sell_dates)}")
        
        return signals
    
    def visualize_strategy(self):
        fig = go.Figure()
        
        # Plot closing price
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Close'],
            name='Close Price',
            line=dict(color='black', width=2)
        ))
        
        # Plot moving averages
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['MA_20'],
            name='20-day MA',
            line=dict(color='orange', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['MA_50'],
            name='50-day MA',
            line=dict(color='green', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['MA_200'],
            name='200-day MA',
            line=dict(color='red', width=1.5)
        ))
        
        # Plot Bollinger Bands
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['BB_Upper'],
            name='BB Upper',
            line=dict(color='blue', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.indicators['BB_Lower'],
            name='BB Lower',
            line=dict(color='blue', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))
        
        # Plot support and resistance levels
        if hasattr(self, 'support_resistance') and self.support_resistance is not None:
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.support_resistance['support'],
                name='Support',
                line=dict(color='green', width=1, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.support_resistance['resistance'],
                name='Resistance',
                line=dict(color='red', width=1, dash='dot')
            ))
        
        # Plot initial buy signals (unconfirmed)
        initial_buy_signals = self.signals[self.signals['signal'] == 1]
        if len(initial_buy_signals) > 0:
            buy_prices = self.data.loc[initial_buy_signals.index, 'Close']
            fig.add_trace(go.Scatter(
                x=initial_buy_signals.index,
                y=buy_prices,
                name=f'Initial Buy Signal ({len(initial_buy_signals)})',
                mode='markers',
                marker=dict(color='yellow', symbol='triangle-up', size=12, opacity=0.6)
            ))
        
        # Plot initial sell signals (unconfirmed)
        initial_sell_signals = self.signals[self.signals['signal'] == -1]
        if len(initial_sell_signals) > 0:
            sell_prices = self.data.loc[initial_sell_signals.index, 'Close']
            fig.add_trace(go.Scatter(
                x=initial_sell_signals.index,
                y=sell_prices,
                name=f'Initial Sell Signal ({len(initial_sell_signals)})',
                mode='markers',
                marker=dict(color='orange', symbol='triangle-down', size=12, opacity=0.6)
            ))
        
        # Plot confirmed buy signals (with S/R validation)
        confirmed_buy_signals = self.signals[self.signals['confirmed_signal'] == 1]
        if len(confirmed_buy_signals) > 0:
            buy_prices = self.data.loc[confirmed_buy_signals.index, 'Close']
            fig.add_trace(go.Scatter(
                x=confirmed_buy_signals.index,
                y=buy_prices,
                name=f'Confirmed Buy Signal ({len(confirmed_buy_signals)})',
                mode='markers',
                marker=dict(color='lime', symbol='triangle-up', size=15)
            ))
        
        # Plot confirmed sell signals (with S/R validation)
        confirmed_sell_signals = self.signals[self.signals['confirmed_signal'] == -1]
        if len(confirmed_sell_signals) > 0:
            sell_prices = self.data.loc[confirmed_sell_signals.index, 'Close']
            fig.add_trace(go.Scatter(
                x=confirmed_sell_signals.index,
                y=sell_prices,
                name=f'Confirmed Sell Signal ({len(confirmed_sell_signals)})',
                mode='markers',
                marker=dict(color='red', symbol='triangle-down', size=15)
            ))
        
        fig.update_layout(
            title=f'{self.ticker} Enhanced Trading Strategy with S/R Validation',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            plot_bgcolor='white',
            paper_bgcolor='lightgray',
            font=dict(color='black')
        )
        
        st.plotly_chart(fig, use_container_width=True)

class RealTimeTradingStrategy:
    def __init__(self, ticker='AAPL', data_source='simulated'):
        self.ticker = ticker
        self.data_source = data_source
        self.real_time_data = pd.DataFrame()
        self.signals = None
        self.indicators = None
        self.support_resistance = None
        self.running = False
        self.update_thread = None
        
        print(f"Initializing Real-Time Trading Strategy for {ticker}")
        print(f"Data Source: {data_source}")
        
        # Initialize with some historical data to start
        self.initialize_with_historical_data()
    
    def initialize_with_historical_data(self):
        """Load initial historical data for the real-time strategy"""
        try:
            print(f"Initializing real-time strategy for {self.ticker}")
            
            # Fetch recent historical data
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(days=30)
            
            print(f"Fetching historical data from {start_date} to {end_date}")
            historical_data = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if historical_data is None or historical_data.empty:
                print("No historical data received, creating sample data")
                # Create sample data if yfinance fails
                sample_dates = pd.date_range(end=end_date, periods=30, freq='B')
                sample_prices = np.linspace(100, 150, 30) + np.random.normal(0, 5, 30)
                
                self.real_time_data = pd.DataFrame({
                    'Open': sample_prices,
                    'High': sample_prices * 1.01,
                    'Low': sample_prices * 0.99,
                    'Close': sample_prices,
                    'Volume': [1000000] * 30
                }, index=sample_dates)
                
                print(f"Created sample data with {len(self.real_time_data)} data points")
            else:
                self.real_time_data = historical_data.copy()
                print(f"Initialized with historical data: {len(self.real_time_data)} data points")
            
            # Calculate initial indicators and signals
            self.calculate_real_time_indicators()
            self.generate_real_time_signals()
            
        except Exception as e:
            print(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            # Create fallback sample data
            self.create_fallback_data()
    
    def create_fallback_data(self):
        """Create fallback sample data when yfinance fails"""
        try:
            print("Creating fallback sample data")
            end_date = pd.Timestamp.today()
            sample_dates = pd.date_range(end=end_date, periods=30, freq='B')
            sample_prices = np.linspace(100, 150, 30) + np.random.normal(0, 5, 30)
            
            self.real_time_data = pd.DataFrame({
                'Open': sample_prices,
                'High': sample_prices * 1.01,
                'Low': sample_prices * 0.99,
                'Close': sample_prices,
                'Volume': [1000000] * 30
            }, index=sample_dates)
            
            print(f"Created fallback data with {len(self.real_time_data)} data points")
        except Exception as e:
            print(f"Fallback data creation failed: {e}")
    
    def fetch_real_time_data_simulated(self):
        """Simulate real-time data for demonstration"""
        try:
            # Use the last available data point from our real_time_data
            if self.real_time_data.empty:
                print("No historical data available for simulation")
                return None
            
            # Get the last available data point
            last_data = self.real_time_data.iloc[-1:].copy()
            last_close = last_data['Close'].iloc[0]
            
            # Generate simulated real-time data
            timestamp = pd.Timestamp.now()
            
            # Simulate price movement with some randomness
            price_change = np.random.normal(0, 0.005)  # ~0.5% daily move
            simulated_close = last_close * (1 + price_change)
            
            simulated_data = pd.DataFrame({
                'Open': [last_close],
                'High': [simulated_close * 1.002],
                'Low': [simulated_close * 0.998],
                'Close': [simulated_close],
                'Volume': [self.real_time_data['Volume'].mean() if 'Volume' in self.real_time_data.columns else 1000000]
            }, index=[timestamp])
            
            return simulated_data
            
        except Exception as e:
            print(f"Real-time data simulation error: {e}")
            return None
    
    def update_with_real_time_data(self, new_data):
        """Update strategy with new real-time data"""
        try:
            if new_data is None or new_data.empty:
                return False
            
            # Append new data
            if self.real_time_data.empty:
                self.real_time_data = new_data
            else:
                # Avoid duplicate timestamps
                if new_data.index[0] not in self.real_time_data.index:
                    self.real_time_data = pd.concat([self.real_time_data, new_data])
                else:
                    # Update existing timestamp
                    self.real_time_data.loc[new_data.index[0]] = new_data.iloc[0]
            
            # Calculate indicators and signals
            self.calculate_real_time_indicators()
            self.generate_real_time_signals()
            
            return True
            
        except Exception as e:
            print(f"Real-time update error: {e}")
            return False
    
    def calculate_real_time_indicators(self):
        """Calculate indicators for real-time data"""
        if self.real_time_data.empty or len(self.real_time_data) < 2:
            return None
        
        close_prices = self.real_time_data['Close'].values
        high_prices = self.real_time_data['High'].values
        low_prices = self.real_time_data['Low'].values
        
        indicators = {}
        
        # Moving Averages (only calculate if we have enough data)
        if len(close_prices) >= 20:
            indicators['MA_20'] = talib.SMA(close_prices, timeperiod=20)
        else:
            indicators['MA_20'] = np.full(len(close_prices), np.nan)
            
        if len(close_prices) >= 50:
            indicators['MA_50'] = talib.SMA(close_prices, timeperiod=50)
        else:
            indicators['MA_50'] = np.full(len(close_prices), np.nan)
            
        if len(close_prices) >= 200:
            indicators['MA_200'] = talib.SMA(close_prices, timeperiod=200)
        else:
            indicators['MA_200'] = np.full(len(close_prices), np.nan)
        
        # RSI (only calculate if we have enough data)
        if len(close_prices) >= 14:
            indicators['RSI'] = talib.RSI(close_prices, timeperiod=14)
        else:
            indicators['RSI'] = np.full(len(close_prices), np.nan)
        
        # Support/Resistance (only calculate if we have enough data)
        support_resistance = pd.DataFrame(index=self.real_time_data.index)
        if len(low_prices) >= 20:
            support_resistance['support'] = pd.Series(low_prices).rolling(window=20).min()
            support_resistance['resistance'] = pd.Series(high_prices).rolling(window=20).max()
        else:
            support_resistance['support'] = np.full(len(low_prices), np.nan)
            support_resistance['resistance'] = np.full(len(high_prices), np.nan)
            
        self.support_resistance = support_resistance
        
        # Convert indicators to DataFrame
        self.indicators = pd.DataFrame(indicators, index=self.real_time_data.index)
        
        return self.indicators
    
    def generate_real_time_signals(self):
        """Generate signals for real-time data"""
        if self.indicators is None or len(self.indicators) < 2:
            return None
        
        signals = pd.DataFrame(index=self.real_time_data.index)
        signals['signal'] = 0
        signals['confirmed_signal'] = 0
        
        # Check if we have valid MA data
        if 'MA_20' not in self.indicators or 'MA_50' not in self.indicators:
            self.signals = signals
            return signals
            
        # Get moving averages
        ma_20 = self.indicators['MA_20']
        ma_50 = self.indicators['MA_50']
        
        # Forward fill NaN values
        ma_20 = ma_20.ffill()
        ma_50 = ma_50.ffill()
        
        # Only generate signals if we have valid MA values
        valid_ma_indices = ma_20.notna() & ma_50.notna()
        
        if valid_ma_indices.sum() < 2:  # Need at least 2 valid points for crossover
            self.signals = signals
            return signals
            
        # Define crossover functions
        def crossover_up(series1, series2):
            curr = series1 > series2
            prev = series1.shift(1) <= series2.shift(1)
            return curr & prev
        
        def crossover_down(series1, series2):
            curr = series1 < series2
            prev = series1.shift(1) >= series2.shift(1)
            return curr & prev
        
        # Generate initial signals
        buy_conditions = crossover_up(ma_20, ma_50)
        sell_conditions = crossover_down(ma_20, ma_50)
        
        signals.loc[buy_conditions, 'signal'] = 1
        signals.loc[sell_conditions, 'signal'] = -1
        
        # Enhanced signal confirmation with support/resistance validation
        # Only do this if we have support/resistance data
        if (self.support_resistance is not None and 
            'support' in self.support_resistance and 
            'resistance' in self.support_resistance):
            
            for i in range(1, len(signals)):
                if signals['signal'].iloc[i] == 1:  # Buy signal
                    try:
                        if (self.real_time_data['Close'].iloc[i] > self.support_resistance['resistance'].iloc[i-1] and
                            self.real_time_data['Close'].iloc[i-1] <= self.support_resistance['resistance'].iloc[i-1]):
                            signals['confirmed_signal'].iloc[i] = 1
                    except (IndexError, KeyError):
                        pass  # Skip if data not available
                        
                elif signals['signal'].iloc[i] == -1:  # Sell signal
                    try:
                        if (self.real_time_data['Close'].iloc[i] < self.support_resistance['support'].iloc[i-1] and
                            self.real_time_data['Close'].iloc[i-1] >= self.support_resistance['support'].iloc[i-1]):
                            signals['confirmed_signal'].iloc[i] = -1
                    except (IndexError, KeyError):
                        pass  # Skip if data not available
        
        self.signals = signals
        return signals
    
    def start_real_time_updates(self):
        """Start real-time data updates in background thread"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._real_time_update_loop, daemon=True)
        self.update_thread.start()
        print("Real-time updates started")
    
    def stop_real_time_updates(self):
        """Stop real-time data updates"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        print("Real-time updates stopped")
    
    def _real_time_update_loop(self):
        """Background loop for real-time updates"""
        while self.running:
            try:
                # Fetch new data (simulated for now)
                new_data = self.fetch_real_time_data_simulated()
                
                if new_data is not None:
                    success = self.update_with_real_time_data(new_data)
                    if success:
                        print(f"Updated with new data at {new_data.index[0]}")
                        # Force Streamlit to rerun to show updates
                        st.experimental_rerun()
                
                # Update every 10 seconds for demo (faster than 60 for testing)
                time.sleep(10)
                
            except Exception as e:
                print(f"Real-time update loop error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def visualize_real_time_strategy(self):
        """Visualize real-time trading strategy"""
        if self.real_time_data.empty:
            print("No real-time data available for visualization")
            return
        
        fig = go.Figure()
        
        # Plot closing price
        fig.add_trace(go.Scatter(
            x=self.real_time_data.index,
            y=self.real_time_data['Close'],
            name='Close Price',
            line=dict(color='black', width=2)
        ))
        
        # Plot moving averages (if available)
        if self.indicators is not None:
            if 'MA_20' in self.indicators:
                fig.add_trace(go.Scatter(
                    x=self.real_time_data.index,
                    y=self.indicators['MA_20'],
                    name='20-day MA',
                    line=dict(color='orange', width=1.5)
                ))
            
            if 'MA_50' in self.indicators:
                fig.add_trace(go.Scatter(
                    x=self.real_time_data.index,
                    y=self.indicators['MA_50'],
                    name='50-day MA',
                    line=dict(color='green', width=1.5)
                ))
        
        # Plot support and resistance levels
        if self.support_resistance is not None:
            fig.add_trace(go.Scatter(
                x=self.real_time_data.index,
                y=self.support_resistance['support'],
                name='Support',
                line=dict(color='green', width=1, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=self.real_time_data.index,
                y=self.support_resistance['resistance'],
                name='Resistance',
                line=dict(color='red', width=1, dash='dot')
            ))
        
        # Plot signals
        if self.signals is not None:
            # Initial signals
            initial_buy_signals = self.signals[self.signals['signal'] == 1]
            if len(initial_buy_signals) > 0:
                buy_prices = self.real_time_data.loc[initial_buy_signals.index, 'Close']
                fig.add_trace(go.Scatter(
                    x=initial_buy_signals.index,
                    y=buy_prices,
                    name=f'Initial Buy Signal ({len(initial_buy_signals)})',
                    mode='markers',
                    marker=dict(color='yellow', symbol='triangle-up', size=12, opacity=0.6)
                ))
            
            initial_sell_signals = self.signals[self.signals['signal'] == -1]
            if len(initial_sell_signals) > 0:
                sell_prices = self.real_time_data.loc[initial_sell_signals.index, 'Close']
                fig.add_trace(go.Scatter(
                    x=initial_sell_signals.index,
                    y=sell_prices,
                    name=f'Initial Sell Signal ({len(initial_sell_signals)})',
                    mode='markers',
                    marker=dict(color='orange', symbol='triangle-down', size=12, opacity=0.6)
                ))
            
            # Confirmed signals
            confirmed_buy_signals = self.signals[self.signals['confirmed_signal'] == 1]
            if len(confirmed_buy_signals) > 0:
                buy_prices = self.real_time_data.loc[confirmed_buy_signals.index, 'Close']
                fig.add_trace(go.Scatter(
                    x=confirmed_buy_signals.index,
                    y=buy_prices,
                    name=f'Confirmed Buy Signal ({len(confirmed_buy_signals)})',
                    mode='markers',
                    marker=dict(color='lime', symbol='triangle-up', size=15)
                ))
            
            confirmed_sell_signals = self.signals[self.signals['confirmed_signal'] == -1]
            if len(confirmed_sell_signals) > 0:
                sell_prices = self.real_time_data.loc[confirmed_sell_signals.index, 'Close']
                fig.add_trace(go.Scatter(
                    x=confirmed_sell_signals.index,
                    y=sell_prices,
                    name=f'Confirmed Sell Signal ({len(confirmed_sell_signals)})',
                    mode='markers',
                    marker=dict(color='red', symbol='triangle-down', size=15)
                ))
        
        fig.update_layout(
            title=f'{self.ticker} Real-Time Trading Strategy',
            xaxis_title='Date/Time',
            yaxis_title='Price ($)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            plot_bgcolor='white',
            paper_bgcolor='lightgray',
            font=dict(color='black')
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title('Advanced Trading Strategy')
    st.sidebar.header('Strategy Selection')
    
    # Strategy Selection
    strategy_type = st.sidebar.radio(
        'Choose Strategy Type:',
        ['Historical Analysis', 'Real-Time Simulation'],
        help='Select between historical backtesting or real-time data simulation'
    )
    
    if strategy_type == 'Historical Analysis':
        st.sidebar.header('Historical Analysis Parameters')
        
        # User Input for Historical Strategy
        ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
        start_date = st.sidebar.text_input('Start Date (YYYY-MM-DD)', '2021-01-01')
        end_date = st.sidebar.text_input('End Date (YYYY-MM-DD)', '')
        
        if st.sidebar.button('Run Historical Strategy'):
            try:
                # Initialize and Run Historical Strategy
                strategy = AdvancedTradingStrategy(
                    ticker=ticker, 
                    start_date=start_date, 
                    end_date=end_date if end_date else None
                )
                
                # Fetch Historical/Projected Data
                data = strategy.fetch_historical_data()
                if data is None or data.empty:
                    st.error('Failed to fetch data. Please check the ticker symbol and try again.')
                    return
                
                # Calculate Indicators
                indicators = strategy.calculate_indicators()
                if not indicators:
                    st.error('Failed to calculate indicators.')
                    return
                
                # Calculate Support/Resistance
                support_resistance = strategy.calculate_support_resistance()
                
                # Generate Signals
                signals = strategy.generate_signals()
                
                # Visualize Strategy
                strategy.visualize_strategy()
                
                # Display Data
                st.subheader('Data')
                st.dataframe(data.tail(20))
                
                # Display Indicators
                st.subheader('Indicators')
                st.dataframe(pd.DataFrame(indicators).tail(20))
                
                # Display Support/Resistance
                st.subheader('Support/Resistance Levels')
                if hasattr(strategy, 'support_resistance') and strategy.support_resistance is not None:
                    sr_display = strategy.support_resistance.copy()
                    sr_display['Close'] = strategy.data['Close']
                    st.dataframe(sr_display.tail(20))
                
                # Display Signals
                st.subheader('Signals')
                signals_df = signals.copy()
                signals_df['Close'] = strategy.data['Close']
                
                # Show both initial and confirmed signals
                st.write('**Initial Signals (MA Crossovers):**')
                initial_signals = signals_df[signals_df['signal'] != 0]
                st.dataframe(initial_signals)
                
                st.write('**Confirmed Signals (with S/R Validation):**')
                confirmed_signals = signals_df[signals_df['confirmed_signal'] != 0]
                st.dataframe(confirmed_signals)
                
                # Summary statistics
                st.write(f"**Signal Summary:**")
                st.write(f"- Initial Buy Signals: {len(initial_signals[initial_signals['signal'] == 1])}")
                st.write(f"- Initial Sell Signals: {len(initial_signals[initial_signals['signal'] == -1])}")
                st.write(f"- Confirmed Buy Signals: {len(confirmed_signals[confirmed_signals['confirmed_signal'] == 1])}")
                st.write(f"- Confirmed Sell Signals: {len(confirmed_signals[confirmed_signals['confirmed_signal'] == -1])}")
                
            except Exception as e:
                st.error(f'An unexpected error occurred: {e}')
                import traceback
                traceback.print_exc()
    
    else:  # Real-Time Simulation
        st.sidebar.header('Real-Time Simulation Parameters')
        
        # User Input for Real-Time Strategy
        ticker = st.sidebar.text_input('Stock Ticker', 'AAPL', key='rt_ticker')
        
        # Real-time controls
        if 'rt_strategy' not in st.session_state:
            st.session_state.rt_strategy = None
            st.session_state.rt_running = False
        
        if st.sidebar.button('Start Real-Time Simulation'):
            if not st.session_state.rt_running:
                # Initialize Real-Time Strategy
                st.session_state.rt_strategy = RealTimeTradingStrategy(ticker=ticker)
                st.session_state.rt_strategy.start_real_time_updates()
                st.session_state.rt_running = True
                st.sidebar.success("Real-time simulation started!")
            else:
                st.sidebar.warning("Real-time simulation is already running!")
        
        if st.sidebar.button('Stop Real-Time Simulation'):
            if st.session_state.rt_running and st.session_state.rt_strategy:
                st.session_state.rt_strategy.stop_real_time_updates()
                st.session_state.rt_running = False
                st.sidebar.info("Real-time simulation stopped.")
        
        # Display Real-Time Data
        if st.session_state.rt_running and st.session_state.rt_strategy:
            strategy = st.session_state.rt_strategy
            
            # Show status
            st.sidebar.info(f"Real-time simulation running for {strategy.ticker}")
            
            if not strategy.real_time_data.empty:
                # Visualize Real-Time Strategy
                strategy.visualize_real_time_strategy()
                
                # Display Real-Time Data
                st.subheader('Real-Time Data')
                st.dataframe(strategy.real_time_data.tail(10))
                
                # Display Indicators
                if strategy.indicators is not None:
                    st.subheader('Real-Time Indicators')
                    st.dataframe(strategy.indicators.tail(10))
                else:
                    st.info("Calculating indicators... Need more data points.")
                
                # Display Signals
                if strategy.signals is not None:
                    st.subheader('Real-Time Signals')
                    signals_df = strategy.signals.copy()
                    signals_df['Close'] = strategy.real_time_data['Close']
                    
                    # Show both initial and confirmed signals
                    st.write('**Initial Signals (MA Crossovers):**')
                    initial_signals = signals_df[signals_df['signal'] != 0]
                    if not initial_signals.empty:
                        st.dataframe(initial_signals)
                    else:
                        st.write("No initial signals detected yet.")
                    
                    st.write('**Confirmed Signals (with S/R Validation):**')
                    confirmed_signals = signals_df[signals_df['confirmed_signal'] != 0]
                    if not confirmed_signals.empty:
                        st.dataframe(confirmed_signals)
                    else:
                        st.write("No confirmed signals detected yet.")
                    
                    # Summary statistics
                    st.write(f"**Real-Time Signal Summary:**")
                    st.write(f"- Initial Buy Signals: {len(initial_signals[initial_signals['signal'] == 1])}")
                    st.write(f"- Initial Sell Signals: {len(initial_signals[initial_signals['signal'] == -1])}")
                    st.write(f"- Confirmed Buy Signals: {len(confirmed_signals[confirmed_signals['confirmed_signal'] == 1])}")
                    st.write(f"- Confirmed Sell Signals: {len(confirmed_signals[confirmed_signals['confirmed_signal'] == -1])}")
                else:
                    st.info("Generating signals... Need more data points.")
                    
                # Show update status
                st.info(f"Last update: {strategy.real_time_data.index[-1]} | Next update in ~10 seconds")
            else:
                st.info("Initializing real-time data... This may take a moment.")
        elif st.session_state.rt_running:
            st.info("Real-time simulation is starting up...")

if __name__ == "__main__":
    main()