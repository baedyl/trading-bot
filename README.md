# Advanced Trading Strategy with Support/Resistance Validation

A comprehensive trading strategy application with both historical analysis and real-time simulation capabilities.

## Features

### Enhanced Trading Strategy
- **Moving Average Crossover Signals**: 20-day MA crossing 50-day MA
- **Support/Resistance Validation**: Confirms signals only when price breaks key levels
- **Multiple Technical Indicators**: RSI, MACD, Bollinger Bands
- **Signal Quality Filtering**: Distinguishes between initial and confirmed signals

### Two Strategy Modes
1. **Historical Analysis**: Backtest strategies on historical data
2. **Real-Time Simulation**: Simulate live market conditions with streaming data

### Visualization
- Interactive Plotly charts
- Clear signal markers (initial vs confirmed)
- Support/resistance level visualization
- Moving averages and Bollinger Bands overlay

## Requirements

- Python 3.7+
- Streamlit
- yfinance
- pandas
- numpy
- plotly
- TA-Lib

## Installation

```bash
# Install Python dependencies
pip install streamlit yfinance pandas numpy plotly TA-Lib

# For TA-Lib installation on macOS
brew install ta-lib

# For TA-Lib installation on Linux
sudo apt-get install build-essential
pip install TA-Lib
```

## Running the Application

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Streamlit application
streamlit run bot.py
```

### Streamlit Cloud Deployment

1. **Create a GitHub repository** with all the files (`bot.py`, `README.md`, `requirements.txt`)
2. **Go to Streamlit Cloud** (https://share.streamlit.io/)
3. **Click "New app"** and connect to your GitHub repository
4. **Configure the app**:
   - Repository: Your GitHub repo
   - Branch: main/master
   - Main file path: `bot.py`
   - Python version: 3.9 or 3.10 (recommended)
5. **Click "Deploy"**

### Common Streamlit Cloud Issues

**ModuleNotFoundError**: Ensure your `requirements.txt` file is complete and in the root directory

**TA-Lib installation**: Streamlit Cloud has TA-Lib pre-installed, but if you have issues:
- Use the exact versions in `requirements.txt`
- Try Python 3.9 instead of newer versions

**App not updating**: Clear cache and restart the app in Streamlit Cloud settings

## Usage Instructions

### Historical Analysis Mode

1. **Select Strategy Type**: Choose "Historical Analysis" from the sidebar
2. **Enter Parameters**:
   - **Stock Ticker**: Enter a valid stock symbol (e.g., "AAPL", "MSFT", "GOOG")
   - **Start Date**: Use format YYYY-MM-DD (e.g., "2023-01-01")
   - **End Date**: Optional - leave blank for today's date
3. **Run Strategy**: Click "Run Historical Strategy"
4. **View Results**:
   - Interactive chart with price action and indicators
   - Data tables showing raw prices, indicators, and signals
   - Signal summary with counts of initial vs confirmed signals

### Real-Time Simulation Mode

1. **Select Strategy Type**: Choose "Real-Time Simulation" from the sidebar
2. **Enter Parameters**:
   - **Stock Ticker**: Enter any stock symbol
3. **Start Simulation**: Click "Start Real-Time Simulation"
4. **Monitor Updates**:
   - Data updates every 10 seconds
   - Watch signals generate in real-time
   - See support/resistance validation in action
5. **Stop Simulation**: Click "Stop Real-Time Simulation" when finished

## Signal Interpretation

### Signal Types

- **Yellow Triangles (▲)**: Initial buy signals (MA crossover only)
- **Orange Triangles (▼)**: Initial sell signals (MA crossover only)
- **Lime Triangles (▲)**: Confirmed buy signals (MA crossover + resistance breakout)
- **Red Triangles (▼)**: Confirmed sell signals (MA crossover + support breakdown)

### Chart Elements

- **Black Line**: Close price
- **Orange Line**: 20-day Moving Average
- **Green Line**: 50-day Moving Average
- **Red Line**: 200-day Moving Average
- **Blue Dashed Lines**: Bollinger Bands
- **Green Dotted Line**: Support level
- **Red Dotted Line**: Resistance level

## Troubleshooting

### Common Issues

**"Failed to fetch data" error:**
- Check your internet connection
- Verify the ticker symbol is correct
- Try a different date range
- Use recent dates (past 5 years work best)

**TA-Lib installation issues:**
- Ensure you have the system dependencies installed
- On Windows, you may need to download TA-Lib binaries
- On Linux/macOS, use the package manager as shown above

**Real-time simulation not updating:**
- Check the console logs for errors
- Ensure no other process is blocking the Streamlit port
- Try refreshing the browser page

## Strategy Logic

### Signal Generation Process

1. **Moving Average Crossover Detection**
   - Buy signal: 20MA crosses above 50MA
   - Sell signal: 20MA crosses below 50MA

2. **Support/Resistance Validation**
   - Buy confirmation: Price breaks above resistance level
   - Sell confirmation: Price breaks below support level

3. **Signal Filtering**
   - Only confirmed signals are considered high-quality
   - Initial signals may be false breakouts

### Support/Resistance Calculation

- **Support**: 20-period rolling low
- **Resistance**: 20-period rolling high
- Dynamically updates with new price data

## Technical Indicators

- **Moving Averages**: 20, 50, 200-day simple moving averages
- **RSI (14-period)**: Relative Strength Index for momentum
- **MACD (12,26,9)**: Moving Average Convergence Divergence
- **Bollinger Bands (20-period)**: Volatility bands with 2 standard deviations

## Risk Management Recommendations

While this application provides signals, always:
- Use proper position sizing (1-2% risk per trade)
- Implement stop-loss orders
- Consider market conditions and news events
- Backtest thoroughly before using real capital
- Consult with a financial advisor

## License

This project is for educational purposes only. Not financial advice.