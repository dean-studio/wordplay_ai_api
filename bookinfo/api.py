from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any
import asyncio
import httpx
from datetime import datetime

from scraper import KyoboBookScraper
from db_manager import DatabaseManager

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

db = DatabaseManager(
    host='wordplayapi.mycafe24.com',
    user='wordplayapi',
    password='Hazbola2021!',
    db='wordplayapi'
)

def mark_book_as_scraped(kyobo_id: str, success: bool = True):
    query = """
    UPDATE book_scraping 
    SET is_scraped = %s, scraped_at = %s 
    WHERE kyobo_id = %s
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return db.execute_query(query, (success, current_time, kyobo_id))

def check_book_exists(kyobo_id: str):
    query = "SELECT 1 FROM kyobo_books WHERE kyobo_id = %s"
    result = db.fetch_one(query, (kyobo_id,))
    return result is not None

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
    if not db.connect():
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        kyobo_id = request.kyobo_id.strip()

        if not kyobo_id:
            raise HTTPException(status_code=400, detail="kyobo_id is required")

        book_url = f"https://product.kyobobook.co.kr/detail/{kyobo_id}"

        book_details = scraper.scrape(book_url)

        if book_details:
            book_details['kyobo_id'] = kyobo_id

            book_exists = check_book_exists(kyobo_id)

            if book_exists:
                if db.update_book(book_details):
                    mark_book_as_scraped(kyobo_id, True)
                    return ScrapeResponse(
                        success=True,
                        kyobo_id=kyobo_id,
                        data=book_details,
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    mark_book_as_scraped(kyobo_id, False)
                    return ScrapeResponse(
                        success=False,
                        kyobo_id=kyobo_id,
                        error="Failed to update book in database",
                        timestamp=datetime.now().isoformat()
                    )
            else:
                isbn = book_details.get('isbn', '')
                title = book_details.get('title', '')

                book_data = {
                    'kyobo_id': kyobo_id,
                    'isbn': isbn,
                    'title': title
                }

                if db.insert_book(book_data) and db.update_book(book_details):
                    mark_book_as_scraped(kyobo_id, True)
                    return ScrapeResponse(
                        success=True,
                        kyobo_id=kyobo_id,
                        data=book_details,
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    mark_book_as_scraped(kyobo_id, False)
                    return ScrapeResponse(
                        success=False,
                        kyobo_id=kyobo_id,
                        error="Failed to insert book into database",
                        timestamp=datetime.now().isoformat()
                    )
        else:
            mark_book_as_scraped(kyobo_id, False)
            return ScrapeResponse(
                success=False,
                kyobo_id=kyobo_id,
                error="Failed to scrape book details",
                timestamp=datetime.now().isoformat()
            )

    except Exception as e:
        mark_book_as_scraped(request.kyobo_id, False)
        return ScrapeResponse(
            success=False,
            kyobo_id=request.kyobo_id,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )
    finally:
        db.close()

@app.get("/scrape/{kyobo_id}", response_model=ScrapeResponse)
async def scrape_book_get(kyobo_id: str):
    request = ScrapeRequest(kyobo_id=kyobo_id)
    return await scrape_book(request)

if __name__ == "__main__":
    print("Server running on: http://35.233.152.5:8000")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
