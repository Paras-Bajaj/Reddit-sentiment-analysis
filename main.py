from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Optional
from datetime import datetime
import logging
from contextlib import asynccontextmanager
import os
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models
class PostSentiment(BaseModel):
    title: str
    sentiment: str
    link: str
    confidence: float
    score: float
    published: Optional[str] = None

class AnalysisResponse(BaseModel):
    subreddit: str
    count: int
    timestamp: str
    summary: Dict[str, int]
    data: List[PostSentiment]

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    logger.info("Starting up Sentiment Analysis API...")
    # Create frontend directory if it doesn't exist
    os.makedirs("frontend", exist_ok=True)
    yield
    logger.info("Shutting down Sentiment Analysis API...")

# Initialize FastAPI app
app = FastAPI(
    title="Reddit Sentiment Analysis API",
    description="Real-time sentiment analysis of Reddit posts using VADER",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Configure CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Templates
templates = Jinja2Templates(directory="frontend")

def analyze_sentiment(text: str) -> Dict[str, any]:
    """Analyze text sentiment and return detailed results"""
    scores = analyzer.polarity_scores(text)
    
    # Determine sentiment category
    compound_score = scores["compound"]
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {
        "sentiment": sentiment,
        "score": compound_score,
        "confidence": abs(compound_score),
        "scores": scores
    }

def fetch_reddit_feed(subreddit: str, limit: int = 20):
    """Fetch RSS feed from Reddit with error handling"""
    try:
        rss_url = f"https://www.reddit.com/r/{subreddit}/new/.rss"
        logger.info(f"Fetching feed from: {rss_url}")
        feed = feedparser.parse(rss_url)
        
        if feed.bozo:
            logger.warning(f"Feed parsing issues for {subreddit}: {feed.bozo_exception}")
        
        return feed
    except Exception as e:
        logger.error(f"Error fetching feed for {subreddit}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch Reddit feed")

@app.get("/api/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/analyze", response_model=AnalysisResponse)
async def analyze_reddit(
    subreddit: str = Query("technology", description="Subreddit name to analyze"),
    limit: int = Query(20, ge=1, le=100, description="Number of posts to analyze (1-100)")
):
    """
    Analyze sentiment of recent posts in a subreddit
    
    - **subreddit**: Name of the subreddit (e.g., "technology", "python")
    - **limit**: Number of recent posts to analyze (max 100)
    """
    try:
        # Fetch and parse RSS feed
        feed = fetch_reddit_feed(subreddit, limit)
        
        if not feed.entries:
            raise HTTPException(
                status_code=404,
                detail=f"No posts found in subreddit '{subreddit}' or subreddit doesn't exist"
            )
        
        results = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        
        # Analyze each post
        for entry in feed.entries[:limit]:
            analysis = analyze_sentiment(entry.title)
            
            results.append({
                "title": entry.title,
                "sentiment": analysis["sentiment"],
                "link": entry.link,
                "confidence": analysis["confidence"],
                "score": analysis["score"],
                "published": entry.get("published", "")
            })
            
            sentiment_counts[analysis["sentiment"]] += 1
        
        return {
            "subreddit": subreddit,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat(),
            "summary": sentiment_counts,
            "data": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error analyzing {subreddit}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/supported-subreddits")
async def get_supported_subreddits():
    """Get list of popular subreddits for suggestions"""
    popular_subreddits = [
        "technology", "programming", "python", "webdev", "machinelearning",
        "science", "news", "worldnews", "gaming", "movies", "music",
        "askscience", "explainlikeimfive", "todayilearned"
    ]
    return {"subreddits": popular_subreddits}

# Serve frontend
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    with open("frontend/index.html", "r") as f:
        html_content = f.read()
    
    # Inject the correct API URL
    html_content = html_content.replace(
        "const API_BASE_URL = 'http://127.0.0.1:8000/api';",
        "const API_BASE_URL = '/api';"
    )
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )