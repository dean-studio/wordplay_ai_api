from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any
import asyncio
import httpx
from datetime import datetime

from main import KyoboBookScraper

app = FastAPI(
    title="Kyobo Book Scraper API",
    description="교보문고 도서 정보 스크래핑 API",
    version="1.0.0"
)

class ScrapeRequest(BaseModel):
    kyobo_id: str

class ScrapeResponse(BaseModel):
    success: bool
    kyobo_id: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str

scraper = KyoboBookScraper()

@app.get("/")
async def root():
    return {
        "message": "Kyobo Book Scraper API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_book(request: ScrapeRequest):
    try:
        kyobo_id = request.kyobo_id.strip()

        if not kyobo_id:
            raise HTTPException(status_code=400, detail="kyobo_id is required")

        book_url = f"https://product.kyobobook.co.kr/detail/{kyobo_id}"

        book_details = scraper.scrape(book_url)

        if book_details:
            book_details['kyobo_id'] = kyobo_id

            return ScrapeResponse(
                success=True,
                kyobo_id=kyobo_id,
                data=book_details,
                timestamp=datetime.now().isoformat()
            )
        else:
            return ScrapeResponse(
                success=False,
                kyobo_id=kyobo_id,
                error="Failed to scrape book details",
                timestamp=datetime.now().isoformat()
            )

    except Exception as e:
        return ScrapeResponse(
            success=False,
            kyobo_id=request.kyobo_id,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@app.get("/scrape/{kyobo_id}", response_model=ScrapeResponse)
async def scrape_book_get(kyobo_id: str):
    request = ScrapeRequest(kyobo_id=kyobo_id)
    return await scrape_book(request)

if __name__ == "__main__":
    uvicorn.run(
        "start:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
