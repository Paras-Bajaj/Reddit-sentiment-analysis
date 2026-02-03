from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="Live Reddit Sentiment API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()

def analyze(text: str):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

@app.get("/analyze")
def analyze_reddit(subreddit: str = "technology", limit: int = 20):
    rss_url = f"https://www.reddit.com/r/{subreddit}/new/.rss"
    feed = feedparser.parse(rss_url)

    results = []
    for entry in feed.entries[:limit]:
        sentiment = analyze(entry.title)
        results.append({
            "title": entry.title,
            "sentiment": sentiment,
            "link": entry.link
        })

    return {
        "subreddit": subreddit,
        "count": len(results),
        "data": results
    }
