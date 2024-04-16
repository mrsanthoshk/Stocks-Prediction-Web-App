import streamlit as st
from datetime import date, datetime, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
STOCKS = ('GOOG', 'AAPL', 'MSFT', 'GME')
PERIOD_DAYS_MIN = 30
PERIOD_DAYS_MAX = 1825  # Approximately 5 years

# Function to load data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to plot raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Opening Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Closing Price", line=dict(color='red')))
    fig.update_layout(title='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Function to predict forecast with Prophet
def predict_forecast(data, period_days):
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period_days)
    forecast = m.predict(future)
    return m, forecast

# Streamlit setup
st.title('Stock Trend Prediction App')

# Sidebar widgets
selected_stock = st.sidebar.selectbox('Select Stock', STOCKS, index=0)
n_years = st.sidebar.slider('Years of Prediction:', 1, 5, 2)
period_days = st.sidebar.slider('Prediction Period (Days):', PERIOD_DAYS_MIN, PERIOD_DAYS_MAX, PERIOD_DAYS_MAX)

# Loading data
data = load_data(selected_stock)

# Displaying raw data
st.subheader('Raw Data')
st.write(data.tail())

# Plotting raw data
st.subheader('Raw Data Visualization')
plot_raw_data(data)

# Predicting forecast
if data.shape[0] < 2:
    st.error("Insufficient data points for prediction. Please choose another stock or adjust the date range.")
else:
    st.success("Data loaded successfully. Proceeding with prediction.")
    m, forecast = predict_forecast(data, period_days)

    # Displaying forecast data
    st.subheader('Forecast Data')
    st.write(forecast.tail())

    # Plotting forecast
    st.write(f'Forecast Plot for {n_years} Years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Plotting forecast components
    st.write("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # Calculating end date of the prediction
    end_date_prediction = datetime.strptime(TODAY, "%Y-%m-%d") + timedelta(days=period_days)
    st.write("End Date of Prediction:", end_date_prediction.strftime("%Y-%m-%d"))
