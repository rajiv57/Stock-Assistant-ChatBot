import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from google.cloud import language_v1
import os

# Set up Google Cloud Natural Language API credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'charming-module-438013-q2-ecf6dfd91be0.json'

# Initialize the Natural Language client
client = language_v1.LanguageServiceClient()


# Function to analyze sentiment using Google Cloud Natural Language API
def analyze_sentiment(text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
    return sentiment.score, sentiment.magnitude


# Function to extract entities using Google Cloud Natural Language API
def analyze_entities(text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_entities(request={'document': document})
    entities = [(entity.name, entity.type_.name) for entity in response.entities]
    return entities


# Function to get the latest stock price with appropriate currency symbol
def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1y')
    if data.empty:
        return "No data available for the ticker."

    # Get currency information from the stock's info
    stock_info = stock.info
    currency = stock_info.get('currency', 'USD')  # Default to USD if currency info is not available

    # Dictionary to map currency codes to symbols
    currency_symbols = {
        'USD': '$',  # US Dollar
        'INR': '₹',  # Indian Rupee
        'EUR': '€',  # Euro
        'GBP': '£',  # British Pound
        'JPY': '¥',  # Japanese Yen
        'CNY': '¥',  # Chinese Yuan
        'CAD': 'C$',  # Canadian Dollar
        'AUD': 'A$',  # Australian Dollar
        # Add other currencies as needed
    }

    # Get the currency symbol, default to $ if the currency is not in the dictionary
    currency_symbol = currency_symbols.get(currency, '$')

    # Return the latest stock price with the correct currency symbol
    return f'{currency_symbol}{data.iloc[-1].Close}'


# Function to calculate Simple Moving Average (SMA)
def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    if data.empty or len(data) < window:
        return "Not enough data to calculate SMA."
    return str(data.rolling(window=window).mean().iloc[-1])


# Function to calculate Exponential Moving Average (EMA)
def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    if data.empty or len(data) < window:
        return "Not enough data to calculate EMA."
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])


# Function to calculate Relative Strength Index (RSI)
def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    if data.empty:
        return "No data available for the ticker."
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14 - 1, adjust=False).mean()
    ema_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs)).iloc[-1])


# Function to calculate Moving Average Convergence Divergence (MACD)
def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    if data.empty:
        return "No data available for the ticker."
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'


# Function to plot stock price for the last year
def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()  # Close the plot after saving to avoid memory issues


# Streamlit UI
st.title('Stock Analysis Tool with Sentiment Analysis')

# User input for stock ticker symbol
ticker = st.text_input('Enter the stock ticker symbol (e.g., AAPL, MSFT):')

# Dropdown menu to select the analysis type
analysis_type = st.selectbox('Select the analysis:',
                             ['Get Stock Price', 'SMA', 'EMA', 'RSI', 'MACD', 'Plot Stock Price', 'Analyze Sentiment',
                              'Analyze Entities'])

if ticker:
    try:
        # Perform stock-related analysis based on user selection
        if analysis_type == 'Get Stock Price':
            stock_price = get_stock_price(ticker)
            st.write(f'Latest Stock Price of {ticker}: {stock_price}')

        elif analysis_type == 'SMA':
            window = st.number_input('Enter window size for SMA:', min_value=1, max_value=365, value=50)
            sma = calculate_SMA(ticker, window)
            st.write(f'Simple Moving Average (SMA) of {ticker} for window {window}: {sma}')

        elif analysis_type == 'EMA':
            window = st.number_input('Enter window size for EMA:', min_value=1, max_value=365, value=50)
            ema = calculate_EMA(ticker, window)
            st.write(f'Exponential Moving Average (EMA) of {ticker} for window {window}: {ema}')

        elif analysis_type == 'RSI':
            rsi = calculate_RSI(ticker)
            st.write(f'Relative Strength Index (RSI) of {ticker}: {rsi}')

        elif analysis_type == 'MACD':
            macd = calculate_MACD(ticker)
            st.write(f'MACD of {ticker}: {macd}')

        elif analysis_type == 'Plot Stock Price':
            plot_stock_price(ticker)
            st.image('stock.png')

    except Exception as e:
        st.error(f"Error: {e}")

# Text input for sentiment or entity analysis
if analysis_type in ['Analyze Sentiment', 'Analyze Entities']:
    text_input = st.text_area('Enter text for analysis (e.g., stock-related news or description):')

    if text_input:
        if analysis_type == 'Analyze Sentiment':
            sentiment_score, sentiment_magnitude = analyze_sentiment(text_input)
            st.write(f'Sentiment Score: {sentiment_score}, Sentiment Magnitude: {sentiment_magnitude}')

        elif analysis_type == 'Analyze Entities':
            entities = analyze_entities(text_input)
            st.write(f'Entities found: {entities}')
    
   
   
          







