import snscrape.modules.twitter as sntwitter
from nltk.sentiment import SentimentIntensityAnalyzer

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

def get_tweets(stock_symbol):
    tweets = []
    query = f"{stock_symbol} lang:en"
    
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append(tweet.rawContent)
    
    return tweets

def analyze_stock_sentiment(stock_symbol):
    tweets = get_tweets(stock_symbol)
    
    if not tweets:
        print("No tweets found.")
        return
    
    bullish_count = 0
    bearish_count = 0
    
    for tweet in tweets:
        sentiment = analyze_sentiment(tweet)
        if sentiment == "Bullish":
            bullish_count += 1
        elif sentiment == "Bearish":
            bearish_count += 1
    
    if bullish_count > bearish_count:
        print(f"The sentiment for {stock_symbol} is bullish.")
    elif bearish_count > bullish_count:
        print(f"The sentiment for {stock_symbol} is bearish.")
    else:
        print(f"The sentiment for {stock_symbol} is neutral.")

# Example usage:
stock_symbol = "FINPIPE"  # Replace with the desired stock symbol
analyze_stock_sentiment(stock_symbol)
