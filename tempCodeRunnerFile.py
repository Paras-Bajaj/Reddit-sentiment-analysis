from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import status
from fastapi.encoders import jsonable_encoder
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Optional, Any, Generator
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import os
import csv
import pandas as pd
from pydantic import BaseModel, Field, validator, HttpUrl
import uvicorn
from dotenv import load_dotenv
import json
import aiohttp
import asyncio
import aiofiles
import uuid
import hashlib
import time
from pydantic import BaseModel, field_validator
from ratelimit import limits, sleep_and_retry
from ratelimit.exception import RateLimitException
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
import httpx
from cachetools import TTLCache
import pickle
from typing_extensions import Literal
import secrets
import bcrypt
from pathlib import Path
import gzip
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# Load environment variables
load_dotenv()

# Configure structured logging with JSON format for production
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, 'request_id', None),
            "subreddit": getattr(record, 'subreddit', None),
            "duration_ms": getattr(record, 'duration_ms', None),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# Setup logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

# File handler with rotation
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)

# Use JSON formatter in production
if os.getenv('ENVIRONMENT') == 'production':
    json_formatter = JSONFormatter()
    console_handler.setFormatter(json_formatter)
    file_handler.setFormatter(json_formatter)
else:
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )
    console_handler.setFormatter(simple_formatter)
    file_handler.setFormatter(simple_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Initialize metrics as None, will be created inside lifespan
REQUESTS_TOTAL = None
REQUEST_DURATION = None
ANALYSIS_COUNTER = None
ACTIVE_REQUESTS = None
CACHE_HITS = None
CACHE_MISSES = None

def init_metrics():
    """Initialize Prometheus metrics"""
    global REQUESTS_TOTAL, REQUEST_DURATION, ANALYSIS_COUNTER, ACTIVE_REQUESTS, CACHE_HITS, CACHE_MISSES
    
    REQUESTS_TOTAL = Counter('sentiment_requests_total', 'Total requests', ['endpoint', 'method', 'status'])
    REQUEST_DURATION = Histogram('sentiment_request_duration_seconds', 'Request duration in seconds', ['endpoint'])
    ANALYSIS_COUNTER = Counter('sentiment_analysis_total', 'Total analyses performed', ['subreddit', 'sentiment'])
    ACTIVE_REQUESTS = Gauge('sentiment_active_requests', 'Active requests')
    CACHE_HITS = Counter('sentiment_cache_hits', 'Cache hits')
    CACHE_MISSES = Counter('sentiment_cache_misses', 'Cache misses')

# Data directory setup
DATA_DIR = Path("data")
CSV_DIR = DATA_DIR / "csv"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = Path("logs")

for directory in [DATA_DIR, CSV_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

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
    link: HttpUrl
    confidence: float = Field(..., ge=0, le=1)
    score: float = Field(..., ge=-1, le=1)
    published: Optional[datetime] = None
    subreddit: str
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, float]] = None
    word_count: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            HttpUrl: lambda v: str(v)
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

class HealthCheck(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    version: str = "2.0.0"
    uptime: float
    dependencies: Dict[str, str]
    metrics: Dict[str, Any]
    disk_usage: Dict[str, float]
    csv_stats: Dict[str, int]

class RateLimitResponse(BaseModel):
    detail: str
    retry_after: Optional[int] = None

class ExportRequest(BaseModel):
    subreddit: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    format: Literal["csv", "json", "excel"] = "csv"

# Dataclass for CSV storage
@dataclass
class CSVRecord:
    id: str
    subreddit: str
    title: str
    sentiment: str
    score: float
    confidence: float
    link: str
    published: Optional[str]
    analyzed_at: str
    word_count: Optional[int]
    pos_score: Optional[float]
    neg_score: Optional[float]
    neu_score: Optional[float]
    request_id: str

# Enhanced Sentiment Analyzer with batch processing
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.cache = TTLCache(maxsize=5000, ttl=3600)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch"""
        results = []
        for text in texts:
            results.append(self.analyze_single(text))
        return results
    
    def analyze_single(self, text: str) -> Dict[str, Any]:
        """Analyze single text with caching"""
        cache_key = hashlib.sha256(text.encode()).hexdigest()[:32]
        
        if cache_key in self.cache:
            CACHE_HITS.inc()
            return self.cache[cache_key]
        
        CACHE_MISSES.inc()
        scores = self.analyzer.polarity_scores(text)
        
        # Enhanced sentiment categorization
        compound_score = scores["compound"]
        
        if compound_score >= 0.5:
            sentiment = SentimentCategory.VERY_POSITIVE
            confidence_level = "Very High"
        elif compound_score >= 0.15:
            sentiment = SentimentCategory.POSITIVE
            confidence_level = "High"
        elif compound_score >= 0.05:
            sentiment = SentimentCategory.POSITIVE
            confidence_level = "Medium"
        elif compound_score <= -0.5:
            sentiment = SentimentCategory.VERY_NEGATIVE
            confidence_level = "Very High"
        elif compound_score <= -0.15:
            sentiment = SentimentCategory.NEGATIVE
            confidence_level = "High"
        elif compound_score <= -0.05:
            sentiment = SentimentCategory.NEGATIVE
            confidence_level = "Medium"
        else:
            sentiment = SentimentCategory.NEUTRAL
            confidence_level = "Medium"
        
        result = {
            "sentiment": sentiment,
            "sentiment_category": sentiment,
            "confidence_level": confidence_level,
            "score": compound_score,
            "confidence": abs(compound_score),
            "scores": scores,
            "magnitude": scores["pos"] + scores["neg"],
            "word_count": len(text.split()),
            "details": {
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"]
            }
        }
        
        self.cache[cache_key] = result
        return result

analyzer = EnhancedSentimentAnalyzer()

# CSV Manager for data persistence
class CSVDataManager:
    def __init__(self, csv_dir: Path):
        self.csv_dir = csv_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        (self.csv_dir / "daily").mkdir(exist_ok=True, parents=True)
        (self.csv_dir / "subreddits").mkdir(exist_ok=True, parents=True)
        (self.csv_dir / "exports").mkdir(exist_ok=True, parents=True)
    
    def get_daily_filename(self) -> Path:
        """Get filename for today's CSV"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.csv_dir / "daily" / f"analyses_{date_str}.csv"
    
    def get_subreddit_filename(self, subreddit: str) -> Path:
        """Get filename for subreddit-specific CSV"""
        return self.csv_dir / "subreddits" / f"{subreddit}.csv"
    
    def save_analysis(self, analysis_data: AnalysisResponse) -> None:
        """Save analysis results to CSV files"""
        try:
            records = []
            for post in analysis_data.data:
                record = CSVRecord(
                    id=post.id,
                    subreddit=post.subreddit,
                    title=post.title[:500],  # Truncate very long titles
                    sentiment=post.sentiment.value,
                    score=post.score,
                    confidence=post.confidence,
                    link=str(post.link),
                    published=post.published.isoformat() if post.published else None,
                    analyzed_at=post.analyzed_at.isoformat(),
                    word_count=post.word_count,
                    pos_score=post.details.get("positive") if post.details else None,
                    neg_score=post.details.get("negative") if post.details else None,
                    neu_score=post.details.get("neutral") if post.details else None,
                    request_id=analysis_data.request_id
                )
                records.append(asdict(record))
            
            # Save to daily file
            daily_file = self.get_daily_filename()
            self._append_to_csv(daily_file, records)
            
            # Save to subreddit-specific file
            subreddit_file = self.get_subreddit_filename(analysis_data.subreddit)
            self._append_to_csv(subreddit_file, records)
            
            logger.info(f"Saved {len(records)} records for subreddit/{analysis_data.subreddit}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis to CSV: {e}")
    
    def _append_to_csv(self, file_path: Path, records: List[Dict]):
        """Append records to CSV file"""
        file_exists = file_path.exists()
        
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSVRecord.__dataclass_fields__.keys())
            
            if not file_exists:
                writer.writeheader()
            
            for record in records:
                writer.writerow(record)
    
    def load_analyses(self, subreddit: Optional[str] = None, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load analyses from CSV with filtering"""
        try:
            if subreddit:
                file_path = self.get_subreddit_filename(subreddit)
                if not file_path.exists():
                    return pd.DataFrame()
                df = pd.read_csv(file_path)
            else:
                # Load from daily files
                dfs = []
                current_date = start_date or datetime.now() - timedelta(days=7)
                end_date = end_date or datetime.now()
                
                while current_date <= end_date:
                    daily_file = self.csv_dir / "daily" / f"analyses_{current_date.strftime('%Y-%m-%d')}.csv"
                    if daily_file.exists():
                        daily_df = pd.read_csv(daily_file)
                        dfs.append(daily_df)
                    current_date += timedelta(days=1)
                
                if not dfs:
                    return pd.DataFrame()
                
                df = pd.concat(dfs, ignore_index=True)
            
            # Convert date columns
            date_columns = ['published', 'analyzed_at']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Filter by date if specified
            if start_date and 'analyzed_at' in df.columns:
                df = df[df['analyzed_at'] >= start_date]
            if end_date and 'analyzed_at' in df.columns:
                df = df[df['analyzed_at'] <= end_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load analyses from CSV: {e}")
            return pd.DataFrame()
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about stored data"""
        try:
            daily_files = list((self.csv_dir / "daily").glob("*.csv"))
            subreddit_files = list((self.csv_dir / "subreddits").glob("*.csv"))
            
            total_records = 0
            for file in daily_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        total_records += sum(1 for _ in f) - 1  # Subtract header
                except:
                    continue
            
            return {
                "daily_files": len(daily_files),
                "subreddit_files": len(subreddit_files),
                "total_records": total_records,
                "total_subreddits": len(subreddit_files)
            }
        except Exception as e:
            logger.error(f"Failed to get CSV stats: {e}")
            return {}

csv_manager = CSVDataManager(CSV_DIR)

# Cache Manager
class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def get_cache_key(self, subreddit: str, limit: int) -> str:
        """Generate cache key"""
        date_hour = datetime.utcnow().strftime('%Y%m%d%H')
        return f"{subreddit}:{limit}:{date_hour}"
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key"""
        safe_key = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json.gz"
    
    async def get(self, cache_key: str) -> Optional[Dict]:
        """Get cached data"""
        cache_file = self.get_cache_path(cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is expired (older than 1 hour)
            if time.time() - cache_file.stat().st_mtime > 3600:
                cache_file.unlink()
                return None
            
            async with aiofiles.open(cache_file, 'rb') as f:
                compressed = await f.read()
                data = gzip.decompress(compressed)
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None
    
    async def set(self, cache_key: str, data: Dict, ttl: int = 3600):
        """Set cache data"""
        cache_file = self.get_cache_path(cache_key)
        
        try:
            json_data = json.dumps(data, default=str)
            compressed = gzip.compress(json_data.encode('utf-8'))
            
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(compressed)
        except Exception as e:
            logger.error(f"Cache write error: {e}")

cache_manager = CacheManager(CACHE_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    logger.info("🚀 Starting Reddit Sentiment Analysis API v2.0")
    init_metrics()
    # Initialize application state
    app.state.startup_time = time.time()
    app.state.request_count = 0
    app.state.active_connections = set()
    
    # Create background task for cache cleanup
    app.state.cleanup_task = asyncio.create_task(cleanup_old_cache())
    
    logger.info("✅ API startup complete")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API...")
    
    # Cancel background tasks
    if hasattr(app.state, 'cleanup_task'):
        app.state.cleanup_task.cancel()
    
    logger.info("✅ Shutdown complete")

async def cleanup_old_cache():
    """Background task to clean up old cache files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            cache_files = list(CACHE_DIR.glob("*.json.gz"))
            current_time = time.time()
            
            deleted = 0
            for cache_file in cache_files:
                # Delete files older than 24 hours
                if current_time - cache_file.stat().st_mtime > 86400:
                    cache_file.unlink()
                    deleted += 1
            
            if deleted:
                logger.info(f"Cleaned up {deleted} old cache files")
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Reddit Sentiment Analysis API",
    description="""
    Professional-grade real-time sentiment analysis of Reddit posts.
    
    ## Features
    - Real-time sentiment analysis using VADER
    - CSV-based data persistence
    - Intelligent caching with TTL
    - Rate limiting and security
    - Comprehensive metrics and monitoring
    - Batch processing capabilities
    - Export functionality (CSV, JSON, Excel)
    
    ## Endpoints
    - `GET /api/analyze` - Analyze subreddit sentiment
    - `GET /api/historical` - Get historical data
    - `GET /api/export` - Export data
    - `GET /api/metrics` - Prometheus metrics
    - `GET /api/health` - Health check
    """,
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "API Support",
        "email": "support@sentimentapi.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)
# Add GZip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure CORS
allowed_origins = json.loads(os.getenv("ALLOWED_ORIGINS", '["http://localhost:3000", "http://127.0.0.1:8000"]'))
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
)

# Prometheus metrics endpoint
Instrumentator().instrument(app).expose(app, endpoint="/api/metrics")

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Templates
templates = Jinja2Templates(directory="frontend")

# Dependencies
def get_request_id(request: Request) -> str:
    """Generate or get request ID from headers"""
    request_id = request.headers.get('X-Request-ID')
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = get_request_id(request)
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # Log request
    logger.info(
        f"Request {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(process_time, 2),
            "client_ip": request.client.host if request.client else None
        }
    )
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(round(process_time, 2))
    
    return response

# Helper Functions
async def fetch_reddit_feed(subreddit: str, limit: int = 20) -> List[Dict]:
    """Fetch Reddit RSS feed with retry logic"""
    max_retries = 3
    timeout = aiohttp.ClientTimeout(total=30)
    
    for attempt in range(max_retries):
        try:
            rss_url = f"https://www.reddit.com/r/{subreddit}/new/.rss?limit={limit}"
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(rss_url, headers={
                    'User-Agent': 'RedditSentimentAnalyzer/2.0 (+https://github.com/yourusername/sentiment-analysis)'
                }) as response:
                    if response.status != 200:
                        if attempt == max_retries - 1:
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"Failed to fetch subreddit {subreddit}"
                            )
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    if feed.bozo:
                        logger.warning(f"Feed parsing issues: {feed.bozo_exception}")
                    
                    return feed.entries[:limit]
                    
        except aiohttp.ClientError as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch Reddit feed after {max_retries} attempts: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable"
                )
            await asyncio.sleep(2 ** attempt)
    
    return []

def calculate_summary_percentage(summary: Dict[str, int], total: int) -> Dict[str, float]:
    """Calculate percentage for each sentiment"""
    if total == 0:
        return {k: 0.0 for k in summary.keys()}
    
    return {k: round((v / total) * 100, 2) for k, v in summary.items()}

# API Endpoints
@app.get("/api/health", response_model=HealthCheck)
async def health_check(request: Request):
    """Comprehensive health check endpoint"""
    uptime = time.time() - app.state.startup_time
    
    # Check dependencies
    dependencies = {
        "reddit_api": "healthy",
        "csv_storage": "healthy",
        "cache": "healthy"
    }
    
    # Test Reddit connectivity
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.reddit.com/r/technology/new/.rss", timeout=5) as response:
                if response.status != 200:
                    dependencies["reddit_api"] = "degraded"
    except:
        dependencies["reddit_api"] = "unhealthy"
    
    # Check disk usage
    try:
        total, used, free = shutil.disk_usage("/")
        disk_usage = {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "usage_percent": round((used / total) * 100, 2)
        }
    except:
        disk_usage = {"error": "Unable to get disk usage"}
    
    # Get CSV stats
    csv_stats = csv_manager.get_stats()
    
    # Determine overall status
    if all(status == "healthy" for status in dependencies.values()):
        overall_status = "healthy"
    elif any(status == "unhealthy" for status in dependencies.values()):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow(),
        "uptime": round(uptime, 2),
        "dependencies": dependencies,
        "metrics": {
            "active_requests": ACTIVE_REQUESTS._value.get(),
            "total_requests": app.state.request_count,
            "cache_hits": CACHE_HITS._value.get(),
            "cache_misses": CACHE_MISSES._value.get()
        },
        "disk_usage": disk_usage,
        "csv_stats": csv_stats
    }

@app.get("/api/analyze", response_model=AnalysisResponse)
async def analyze_reddit(
    request: Request,
    subreddit: str = Query("technology", description="Subreddit name to analyze"),
    limit: int = Query(20, ge=1, le=200, description="Number of posts to analyze (1-200)"),
    force_refresh: bool = Query(False, description="Force cache refresh"),
    detailed: bool = Query(False, description="Return detailed sentiment scores")
):
    """
    Analyze sentiment of recent posts in a subreddit
    
    Returns comprehensive sentiment analysis with caching support.
    """
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        # Validate subreddit
        if not subreddit.isalnum() and '_' not in subreddit and '-' not in subreddit:
            raise HTTPException(
                status_code=400,
                detail="Invalid subreddit name. Use alphanumeric characters, underscores, or hyphens."
            )
        
        # Check cache
        cache_key = cache_manager.get_cache_key(subreddit, limit)
        if not force_refresh:
            cached_data = await cache_manager.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit for {subreddit}")
                cached_data["cached"] = True
                cached_data["processing_time_ms"] = 0
                ACTIVE_REQUESTS.dec()
                return AnalysisResponse(**cached_data)
        
        # Fetch data
        entries = await fetch_reddit_feed(subreddit, limit)
        
        if not entries:
            raise HTTPException(
                status_code=404,
                detail=f"No posts found in subreddit '{subreddit}' or subreddit doesn't exist"
            )
        
        # Analyze posts
        results = []
        sentiment_counts = {sentiment.value: 0 for sentiment in SentimentCategory}
        
        # Extract titles for batch processing
        titles = [entry.title for entry in entries]
        analyses = analyzer.analyze_batch(titles)
        
        for entry, analysis in zip(entries, analyses):
            post_sentiment = PostSentiment(
                title=entry.title,
                sentiment=analysis["sentiment"],
                link=entry.link,
                confidence=analysis["confidence"],
                score=analysis["score"],
                published=datetime.fromisoformat(entry.published) if hasattr(entry, 'published') else None,
                subreddit=subreddit,
                word_count=analysis["word_count"],
                details=analysis["details"] if detailed else None
            )
            
            results.append(post_sentiment)
            sentiment_counts[analysis["sentiment"].value] += 1
            ANALYSIS_COUNTER.labels(subreddit=subreddit, sentiment=analysis["sentiment"].value).inc()
        
        # Prepare response
        processing_time_ms = (time.time() - start_time) * 1000
        total_posts = len(results)
        
        response_data = {
            "request_id": get_request_id(request),
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
                "posts_returned": len(results),
                "user_agent": request.headers.get("user-agent", ""),
                "client_ip": request.client.host if request.client else None
            }
        }
        
        # Cache the response
        await cache_manager.set(cache_key, response_data)
        
        # Save to CSV in background
        asyncio.create_task(
            csv_manager.save_analysis(AnalysisResponse(**response_data))
        )
        
        # Update metrics
        REQUESTS_TOTAL.labels(endpoint="/api/analyze", method="GET", status="200").inc()
        REQUEST_DURATION.labels(endpoint="/api/analyze").observe(processing_time_ms / 1000)
        
        return AnalysisResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error for {subreddit}: {e}", exc_info=True)
        REQUESTS_TOTAL.labels(endpoint="/api/analyze", method="GET", status="500").inc()
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
    finally:
        ACTIVE_REQUESTS.dec()
        app.state.request_count += 1

@app.get("/api/historical")
async def get_historical_data(
    request: Request,
    subreddit: Optional[str] = Query(None, description="Filter by subreddit"),
    days: int = Query(7, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum records to return")
):
    """Get historical analysis data"""
    try:
        start_date = datetime.now() - timedelta(days=days)
        
        df = csv_manager.load_analyses(
            subreddit=subreddit,
            start_date=start_date
        )
        
        if df.empty:
            return {"message": "No historical data found", "data": []}
        
        # Apply limit
        df = df.head(limit)
        
        # Convert to dict
        data = df.to_dict(orient='records')
        
        # Calculate statistics
        stats = {
            "total_records": len(data),
            "date_range": {
                "start": df['analyzed_at'].min().isoformat() if 'analyzed_at' in df.columns else None,
                "end": df['analyzed_at'].max().isoformat() if 'analyzed_at' in df.columns else None
            },
            "subreddits": df['subreddit'].nunique() if 'subreddit' in df.columns else 0,
            "sentiment_distribution": df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {}
        }
        
        return {
            "stats": stats,
            "data": data
        }
        
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve historical data")

@app.post("/api/export")
async def export_data(
    request: Request,
    export_request: ExportRequest
):
    """Export analysis data in various formats"""
    try:
        # Load data
        df = csv_manager.load_analyses(
            subreddit=export_request.subreddit,
            start_date=export_request.start_date,
            end_date=export_request.end_date
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for export")
        
        # Prepare filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subreddit_part = f"_{export_request.subreddit}" if export_request.subreddit else ""
        filename = f"sentiment_analysis{subreddit_part}_{timestamp}"
        
        if export_request.format == "csv":
            # Stream CSV response
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}.csv"
                }
            )
            
        elif export_request.format == "json":
            # JSON response
            data = df.to_dict(orient='records')
            response = JSONResponse(
                content={
                    "metadata": {
                        "exported_at": datetime.utcnow().isoformat(),
                        "record_count": len(data),
                        "format": "json"
                    },
                    "data": data
                }
            )
            response.headers["Content-Disposition"] = f"attachment; filename={filename}.json"
            
        elif export_request.format == "excel":
            # Excel response
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Analysis Data')
            
            response = StreamingResponse(
                io.BytesIO(output.getvalue()),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}.xlsx"
                }
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")

@app.get("/api/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/api/supported-subreddits")
async def get_supported_subreddits():
    """Get list of popular subreddits with metadata"""
    popular_subreddits = [
        {"name": "technology", "category": "Technology", "subscribers": "10M+"},
        {"name": "programming", "category": "Technology", "subscribers": "5M+"},
        {"name": "python", "category": "Technology", "subscribers": "3M+"},
        {"name": "webdev", "category": "Technology", "subscribers": "1M+"},
        {"name": "machinelearning", "category": "Technology", "subscribers": "2M+"},
        {"name": "science", "category": "Science", "subscribers": "30M+"},
        {"name": "news", "category": "News", "subscribers": "35M+"},
        {"name": "worldnews", "category": "News", "subscribers": "33M+"},
        {"name": "gaming", "category": "Entertainment", "subscribers": "38M+"},
        {"name": "movies", "category": "Entertainment", "subscribers": "30M+"},
        {"name": "music", "category": "Entertainment", "subscribers": "32M+"},
        {"name": "askscience", "category": "Education", "subscribers": "24M+"},
        {"name": "explainlikeimfive", "category": "Education", "subscribers": "22M+"},
        {"name": "todayilearned", "category": "Education", "subscribers": "34M+"},
        {"name": "personalfinance", "category": "Finance", "subscribers": "19M+"},
        {"name": "investing", "category": "Finance", "subscribers": "2M+"},
        {"name": "cryptocurrency", "category": "Finance", "subscribers": "6M+"},
        {"name": "fitness", "category": "Health", "subscribers": "12M+"},
        {"name": "nutrition", "category": "Health", "subscribers": "3M+"},
        {"name": "travel", "category": "Lifestyle", "subscribers": "15M+"},
    ]
    
    # Add recently analyzed subreddits from CSV
    try:
        subreddit_files = list((CSV_DIR / "subreddits").glob("*.csv"))
        analyzed_subreddits = [f.stem for f in subreddit_files if f.stem not in [s["name"] for s in popular_subreddits]]
        
        for subreddit in analyzed_subreddits[:10]:  # Limit to 10 recent
            popular_subreddits.append({
                "name": subreddit,
                "category": "Recently Analyzed",
                "subscribers": "N/A"
            })
    except:
        pass
    
    return {"subreddits": popular_subreddits}

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    csv_stats = csv_manager.get_stats()
    
    # Get cache statistics
    cache_files = list(CACHE_DIR.glob("*.json.gz"))
    cache_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)  # MB
    
    return {
        "csv_storage": csv_stats,
        "cache": {
            "files": len(cache_files),
            "size_mb": round(cache_size, 2),
            "hits": CACHE_HITS._value.get(),
            "misses": CACHE_MISSES._value.get(),
            "hit_ratio": round(CACHE_HITS._value.get() / max(CACHE_HITS._value.get() + CACHE_MISSES._value.get(), 1), 2)
        },
        "requests": {
            "total": app.state.request_count,
            "active": ACTIVE_REQUESTS._value.get()
        },
        "uptime_seconds": round(time.time() - app.state.startup_time, 2)
    }

# Serve frontend
@app.get("/")
async def serve_frontend(request: Request):
    """Serve the main frontend page"""
    try:
        with open("frontend/index.html", "r", encoding='utf-8') as f:
            html_content = f.read()
        
        # Inject API URL
        api_url = str(request.url).replace(request.url.path, '') + "/api"
        html_content = html_content.replace(
            "const API_BASE_URL = '/api';",
            f"const API_BASE_URL = '{api_url}';"
        )
        
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"detail": "Frontend not found. Please build the frontend."}
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={
            "request_id": getattr(request.state, 'request_id', None),
            "path": request.url.path,
            "status_code": exc.status_code
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": getattr(request.state, 'request_id', None),
            "timestamp": datetime.utcnow().isoformat()
        },
        headers=exc.headers if hasattr(exc, 'headers') else None
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={
            "request_id": getattr(request.state, 'request_id', None),
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": getattr(request.state, 'request_id', None),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import shutil
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") != "production",
        log_level=log_level.lower(),
        access_log=False,  # We handle logging ourselves
        workers=int(os.getenv("WORKERS", 1))
    )