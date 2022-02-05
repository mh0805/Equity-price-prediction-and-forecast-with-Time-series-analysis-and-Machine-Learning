# Libraries needed: streamlit, pystan, prophet, yfinance, plotly
# pip install streamlit
# pip install pystan
# pip install prophet
# pip install yfinance
# pip install plotly

# Codes inspired by/referenced from/modified from:
# Credit where it is due: "Build A Stock Prediction Web App in Python" - Python Engineer on YouTube
# Link: https://www.youtube.com/watch?v=0E_31WqVzCY&t=518s

# Imports
import streamlit as st
from datetime import date
import yfinance as yf

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Setting Dates
start_date = '2011-01-01'
today_date = date.today().strftime("%Y-%m-%d")

# Title of the app
st.title('Stock Price Prediction')

# Predetermined tickers. Can add to the list upon request, given the data is available on Yahoo Finance. 
stocks = ('AAPL', 'AMC', 'AMZN', 'BA', 'BABA', 'FB', 'GME', 'GOOG', 'MSFT', 'NFLX', 'NVDA', 'QQQ', 'SPY', 'TSLA', 'VIX', '^IXIC', '^GSPC', '^DJI', '^RUT')
selected_stocks = st.selectbox('Select dataset for prediction', stocks)

n_months = st.slider('Prediction Period (Months):', 1, 6)
period = n_months * 31

# Cacheing the data, so the app has data available.
@st.cache
# Defining function to download historical stock data from Yahoo Finance.
def stock_data(ticker):
    data = yf.download(ticker, start_date, today_date)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Fetching historical stock data from Yahoo Finance...')
data = stock_data(selected_stocks)
data_load_state.text('Fetching historical stock data from Yahoo Finance... completed!')

# Visualizing the raw dataframe
st.subheader('Historical Stock Data')
# Visualizing the last 10 trading days data
st.write(data)

# Recent 10 trading days data
st.subheader('Recent Stock Data')
st.write(data.tail(10))

# Defining a function to visualize data
def plot_stock_data():
    # Defining the plot size
    layout = go.Layout(autosize=False, width=1000, height=600)
    # Defining figure
    fig = go.Figure(layout = layout)
    # X and Y axes labels
    fig.update_xaxes(title_text = 'Time')
    fig.update_yaxes(title_text = 'Price ($)')
    # OHLC + Adj Close plots
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open_Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close_Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name='High'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name='Low'))    
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close'], name='Adjusted_Close_Price'))
    fig.layout.update(title_text='Time Series Stock Data', xaxis_rangeslider_visible=True)
    
    # Plotting using Plotly graph
    st.plotly_chart(fig)

# Plotting stock data
plot_stock_data()

# Forecasting using Prophet - Forecasting the adjusted closing price. 
df_train = data[['Date', 'Adj Close']]
# Renaming to be compatible with Prophet. X needs to be 'ds' and Y needs to be 'y'
df_train = df_train.rename(columns={'Date': "ds", "Adj Close": "y"})

# Instantiate Prophet model
model = Prophet()
# Fitting training data
model.fit(df_train)
# Future dataframe 
future = model.make_future_dataframe(periods = period)
# Forecasting
forecast = model.predict(future)
# Showing the last few rows of forecast data
st.subheader('Forecast Data using Prophet')
st.write(forecast.tail())

# Plotting the forecasted data using Plotly
st.write('Forecast Stock Graph')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# This is not the plotly chart
st.write('Forecast components')
fig2 = model.plot_components(forecast)
st.write(fig2)