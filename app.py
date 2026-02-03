from ntscraper import Nitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# =========================
# CONFIG
# =========================
QUERY = "Python programming"
MAX_TWEETS = 20

# =========================
# MANUAL NITTER INSTANCES
# =========================
instances = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://nitter.privacydev.net"
]

scraper = Nitter(instances=instances)

# =========================
# FETCH TWEETS
# =========================
print("\nFetching tweets...\n")

data = scraper.get_tweets(
    QUERY,
    mode="term",
    number=MAX_TWEETS
)

tweets = [t["text"] for t in data["tweets"]]

# =========================
# SENTIMENT ANALYSIS
# =========================
analyzer = SentimentIntensityAnalyzer()
results = []

print("Analyzing tweets...\n")

for i, text in enumerate(tweets, start=1):
    score = analyzer.polarity_scores(text)
    compound = score["compound"]

    if compound >= 0.05:
        sentiment = "Positive 😊"
    elif compound <= -0.05:
        sentiment = "Negative 😠"
    else:
        sentiment = "Neutral 😐"

    print(f"Tweet {i}:")
    print(text)
    print(f"Sentiment: {sentiment}")
    print("-" * 80)

    results.append({
        "Tweet": text,
        "Sentiment": sentiment,
        "Compound": compound
    })

# =========================
# DATAFRAME
# =========================
df = pd.DataFrame(results)

print("\nSentiment Summary:\n")
print(df["Sentiment"].value_counts())

df.to_csv("twitter_sentiment_fixed.csv", index=False)
print("\nSaved to twitter_sentiment_fixed.csv")
