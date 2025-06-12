import requests
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from db_manager import DatabaseManager
from main import KyoboBookScraper

db = DatabaseManager(
    host='wordplayapi.mycafe24.com',
    user='wordplayapi',
    password='Hazbola2021!',
    db='wordplayapi'
)

scraper = KyoboBookScraper()

def get_books_without_preview():
    """preview_images가 null인 도서 목록 조회 (상위 10개)"""
    query = """
    SELECT id, kyobo_id, title 
    FROM kyobo_books 
    WHERE preview_images IS NULL
    ORDER BY id
    """

    try:
        books = db.fetch_all(query)
        print(f"📋 preview_images가 없는 도서 상위 {len(books)}건 조회")
        return books
    except Exception as e:
        print(f"❌ 도서 목록 조회 실패: {e}")
        return []

def update_preview_images(book_id, preview_images):
    """preview_images 컬럼 업데이트"""
    query = """
    UPDATE kyobo_books 
    SET preview_images = %s, 
        updated_at = NOW()
    WHERE id = %s
    """

    try:
        preview_json = json.dumps(preview_images, ensure_ascii=False)
        db.execute_query(query, (preview_json, book_id))
        print(f"✅ preview_images 업데이트 완료: book_id={book_id}")
        return True
    except Exception as e:
        print(f"❌ preview_images 업데이트 실패: book_id={book_id}: {e}")
        return False

def process_single_book(kyobo_id, book_id, title):
    """단일 도서 처리"""
    book_url = f"https://product.kyobobook.co.kr/detail/{kyobo_id}"
    print(f"📖 Processing: {title} (kyobo_id: {kyobo_id})")

    try:
        book_details = scraper.scrape(book_url)

        if book_details:
            preview_images = book_details.get('preview', [])

            if not preview_images:
                print(f"⚠️  미리보기 이미지 없음: {title}")
                preview_images = []
            else:
                print(f"📸 미리보기 이미지 {len(preview_images)}개 발견")

            success = update_preview_images(book_id, preview_images)
            return success

        else:
            print(f"❌ 교보문고 상세 정보 스크래핑 실패: {kyobo_id}")
            update_preview_images(book_id, [])
            return False

    except Exception as e:
        print(f"❌ 도서 처리 중 오류 발생: {kyobo_id}: {e}")
        update_preview_images(book_id, [])
        return False

def process_all_books():
    """모든 도서 처리"""
    books = get_books_without_preview()

    if not books:
        print("🎉 처리할 도서가 없습니다.")
        return

    success_count = 0
    total_count = len(books)

    for i, book in enumerate(books, 1):
        book_id = book['id']
        kyobo_id = book['kyobo_id']
        title = book['title']

        print(f"\n[{i}/{total_count}] 처리 중...")

        success = process_single_book(kyobo_id, book_id, title)
        if success:
            success_count += 1

        time.sleep(1)

    print(f"\n🎯 처리 완료: {success_count}/{total_count} 성공")

def process_specific_book(kyobo_id):
    """특정 kyobo_id로 도서 처리"""
    query = """
    SELECT id, kyobo_id, title 
    FROM kyobo_books 
    WHERE kyobo_id = %s
    """

    try:
        book = db.fetch_one(query, (kyobo_id,))
        if book:
            success = process_single_book(book['kyobo_id'], book['id'], book['title'])
            return success
        else:
            print(f"❌ kyobo_id '{kyobo_id}' 도서를 찾을 수 없습니다.")
            return False
    except Exception as e:
        print(f"❌ 특정 도서 처리 실패: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        kyobo_id = sys.argv[1]
        print(f"🎯 특정 도서 처리: {kyobo_id}")
        process_specific_book(kyobo_id)
    else:
        print("🚀 전체 도서 미리보기 이미지 업데이트 시작")
        process_all_books()
