import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import snscrape.modules.twitter as sntwitter
from nltk.sentiment import SentimentIntensityAnalyzer


start_date = "2015-01-01"
end_date = "2021-09-30"

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
stock_data = yf.download(user_input, start=start_date, end=end_date)
df = pd.DataFrame(stock_data)


data = stock_data[['Close']].values

st.subheader('Data from 2015 - 2021')
st.write(df)

st.subheader('Closing Price vs Moving Average')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
ma100 = df.Close.rolling(100).mean()
plt.plot(ma100,'r')
st.pyplot(fig)

# Scale the data using a MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

sequence_length = 60 # number of past days to consider
num_features = 1 # only using the "Close" column

# Split the data into training and testing sets
split_ratio = 0.8
train_size = int(len(data) * split_ratio)
train_data = data[:train_size, :]
test_data = data[train_size-sequence_length:, :]

# Create the input sequences and labels for the training set
X_train, y_train = [], []
for i in range(sequence_length, len(train_data)):
    X_train.append(train_data[i-sequence_length:i, 0:num_features])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Create the input sequences and labels for the testing set
X_test, y_test = [], []
for i in range(sequence_length, len(test_data)):
    X_test.append(test_data[i-sequence_length:i, 0:num_features])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

model = load_model('C:/Users/Matt/stock_prediction/keras_model.h5')

predictions = model.predict(X_test)

# Inverse the scaling of the data
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

st.subheader('Predicted stock prices')
fig0 = plt.figure(figsize = (12,6))
plt.plot(predictions, label="Predicted")
plt.legend()
st.pyplot(fig0)

st.subheader('Actual vs Predicted stock prices')
# Plot the predicted vs actual stock prices
fig1 = plt.figure(figsize = (12,6))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
st.pyplot(fig1)

def analyze_sentiment(tweet):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(tweet)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        return "Bullish"
    elif compound_score <= -0.05:
        return "Bearish"
    else:
        return "Neutral"

limit = 100

def get_tweets(user_input):
    tweets = []
    query = f"{user_input} lang:en"
    
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append(tweet.rawContent)

        if len(tweets) >= limit:
            break
    
    return tweets

def analyze_stock_sentiment(user_input):
    tweets = get_tweets(user_input)
    
    if not tweets:
        print("No tweets found.")
        return
    
    bullish_count = 0
    bearish_count = 0
    
    limit =10

    for tweet in tweets:
        sentiment = analyze_sentiment(tweet)
        if sentiment == "Bullish":
            bullish_count += 1
        elif sentiment == "Bearish":
            bearish_count += 1
    
    
    if bullish_count > bearish_count:
        st.subheader(f"The sentiment for {user_input} is good.")
    elif bearish_count > bullish_count:
        st.subheader(f"The sentiment for {user_input} is bad.")
    else:
        st.subheader(f"The sentiment for {user_input} is neutral.")

analyze_stock_sentiment(user_input)