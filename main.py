from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from contextlib import asynccontextmanager
import os
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
import aiohttp
import uuid
import time
from pydantic import field_validator
from enum import Enum
import traceback
# Load environment variables
load_dotenv()

# Setup logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

simple_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)
console_handler.setFormatter(simple_formatter)
logger.addHandler(console_handler)

# Enum for sentiment categories
class SentimentCategory(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    VERY_POSITIVE = "Very Positive"
    VERY_NEGATIVE = "Very Negative"

# Pydantic Models
class AnalysisRequest(BaseModel):
    subreddit: str = Field(..., min_length=1, max_length=50, description="Subreddit name")
    limit: int = Field(20, ge=1, le=200, description="Number of posts to analyze (1-200)")
    force_refresh: bool = Field(False, description="Force cache refresh")
    detailed: bool = Field(False, description="Return detailed sentiment scores")
    
    @field_validator('subreddit')
    @classmethod
    def validate_subreddit(cls, v: str) -> str:
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Subreddit name must be alphanumeric, underscore, or hyphen')
        return v.lower()

class PostSentiment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    sentiment: SentimentCategory
    link: str
    confidence: float = Field(..., ge=0, le=1)
    score: float = Field(..., ge=-1, le=1)
    published: Optional[datetime] = None
    subreddit: str
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, float]] = None
    word_count: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AnalysisResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subreddit: str
    count: int
    timestamp: datetime
    summary: Dict[str, int]
    summary_percentage: Dict[str, float]
    data: List[PostSentiment]
    cached: bool = False
    processing_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Enhanced Sentiment Analyzer
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch"""
        results = []
        for text in texts:
            results.append(self.analyze_single(text))
        return results
    
    def analyze_single(self, text: str) -> Dict[str, Any]:
        """Analyze single text"""
        scores = self.analyzer.polarity_scores(text)
        compound_score = scores["compound"]
        
        # Determine sentiment
        if compound_score >= 0.5:
            sentiment = SentimentCategory.VERY_POSITIVE
        elif compound_score >= 0.15:
            sentiment = SentimentCategory.POSITIVE
        elif compound_score >= 0.05:
            sentiment = SentimentCategory.POSITIVE
        elif compound_score <= -0.5:
            sentiment = SentimentCategory.VERY_NEGATIVE
        elif compound_score <= -0.15:
            sentiment = SentimentCategory.NEGATIVE
        elif compound_score <= -0.05:
            sentiment = SentimentCategory.NEGATIVE
        else:
            sentiment = SentimentCategory.NEUTRAL
        
        return {
            "sentiment": sentiment,
            "score": compound_score,
            "confidence": abs(compound_score),
            "scores": scores,
            "word_count": len(text.split()),
            "details": {
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"]
            }
        }

analyzer = EnhancedSentimentAnalyzer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    logger.info("Starting Reddit Sentiment Analysis API v2.0")
    app.state.startup_time = time.time()
    app.state.request_count = 0
    
    yield
    
    logger.info("Shutting down API...")

# Initialize FastAPI app
app = FastAPI(
    title="Reddit Sentiment Analysis API",
    description="Real-time sentiment analysis of Reddit posts",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

async def fetch_reddit_feed(subreddit: str, limit: int = 20) -> List[Dict]:
    """Fetch Reddit RSS feed"""
    try:
        rss_url = f"https://www.reddit.com/r/{subreddit}/new/.rss?limit={limit}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(rss_url, headers={
                'User-Agent': 'RedditSentimentAnalyzer/1.0'
            }) as response:
                if response.status != 200:
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                return feed.entries[:limit]
                
    except Exception as e:
        logger.error(f"Failed to fetch Reddit feed: {e}")
        return []

def calculate_summary_percentage(summary: Dict[str, int], total: int) -> Dict[str, float]:
    if total == 0:
        return {k: 0.0 for k in summary.keys()}
    
    return {k: round((v / total) * 100, 2) for k, v in summary.items()}

# API Endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "uptime": round(time.time() - app.state.startup_time, 2)
    }

@app.get("/api/analyze")
async def analyze_reddit(
    subreddit: str = Query("technology", description="Subreddit name"),
    limit: int = Query(20, ge=1, le=100, description="Number of posts"),
    detailed: bool = Query(False, description="Detailed analysis")
):
    """Analyze subreddit sentiment"""
    try:
        start_time = time.time()
        
        # Fetch data
        entries = await fetch_reddit_feed(subreddit, limit)
        
        if not entries:
            raise HTTPException(
                status_code=404,
                detail=f"No posts found in subreddit '{subreddit}'"
            )
        
        # Analyze posts
        results = []
        sentiment_counts = {sentiment.value: 0 for sentiment in SentimentCategory}
        
        titles = [entry.title for entry in entries]
        analyses = analyzer.analyze_batch(titles)
        
        for entry, analysis in zip(entries, analyses):
            try:
                published = None
                if hasattr(entry, 'published'):
                    try:
                        published = datetime.fromisoformat(entry.published.replace('Z', '+00:00'))
                    except:
                        published = datetime.utcnow()
                
                post_sentiment = PostSentiment(
                    title=entry.title,
                    sentiment=analysis["sentiment"],
                    link=entry.link,
                    confidence=analysis["confidence"],
                    score=analysis["score"],
                    published=published,
                    subreddit=subreddit,
                    word_count=analysis["word_count"],
                    details=analysis["details"] if detailed else None
                )
                
                results.append(post_sentiment)
                sentiment_counts[analysis["sentiment"].value] += 1
                    
            except Exception as e:
                logger.error(f"Error processing post: {e}")
                continue
        
        processing_time_ms = (time.time() - start_time) * 1000
        total_posts = len(results)
        
        response_data = {
            "request_id": str(uuid.uuid4()),
            "subreddit": subreddit,
            "count": total_posts,
            "timestamp": datetime.utcnow(),
            "summary": sentiment_counts,
            "summary_percentage": calculate_summary_percentage(sentiment_counts, total_posts),
            "data": results,
            "cached": False,
            "processing_time_ms": round(processing_time_ms, 2),
            "metadata": {
                "limit_requested": limit,
                "posts_returned": len(results)
            }
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        app.state.request_count += 1

@app.get("/api/supported-subreddits")
async def get_supported_subreddits():
    """Get list of popular subreddits"""
    popular_subreddits = [
        {"name": "technology", "category": "Technology"},
        {"name": "python", "category": "Technology"},
        {"name": "programming", "category": "Technology"},
        {"name": "science", "category": "Science"},
        {"name": "news", "category": "News"},
        {"name": "worldnews", "category": "News"},
        {"name": "gaming", "category": "Entertainment"},
        {"name": "movies", "category": "Entertainment"},
        {"name": "music", "category": "Entertainment"},
        {"name": "askscience", "category": "Education"},
        {"name": "explainlikeimfive", "category": "Education"},
        {"name": "todayilearned", "category": "Education"},
    ]
    
    return {"subreddits": popular_subreddits}

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    try:
        with open("frontend/index.html", "r", encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"detail": "Frontend not found"}
        )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    """Handle all exceptions"""
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
# Add this to the end of your main.py file:
# Add this at the very end of your main.py file:

if __name__ == "__main__":
    # For Railway deployment
    import os
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",  # Note: string format for Railway
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1  # Single worker for Railway's free tier
    )