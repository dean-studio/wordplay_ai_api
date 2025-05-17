import requests
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

from db_manager import DatabaseManager
from main import KyoboBookScraper

# 기존 DB 관리자 사용
db = DatabaseManager(
    host='wordplayapi.mycafe24.com',
    user='wordplayapi',
    password='Hazbola2021!',
    db='wordplayapi'
)


def get_unscraped_books(limit=50):
    """스크래핑되지 않은 도서 ID 목록 가져오기"""
    # 새로운 커서 사용
    cursor = db.conn.cursor()
    try:
        query = """
        SELECT kyobo_id 
        FROM book_scraping 
        WHERE is_scraped = FALSE 
        LIMIT %s
        """
        cursor.execute(query, (limit,))
        result = cursor.fetchall()
        return [row[0] for row in result] if result else []
    finally:
        cursor.close()


def mark_book_as_scraped(kyobo_id, success=True):
    """도서를 스크래핑 완료로 표시"""
    cursor = db.conn.cursor()
    try:
        query = """
        UPDATE book_scraping 
        SET is_scraped = %s, scraped_at = %s 
        WHERE kyobo_id = %s
        """
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(query, (success, current_time, kyobo_id))
        db.conn.commit()
        return True
    except Exception as e:
        db.conn.rollback()
        print(f"도서 상태 업데이트 오류: {e}")
        return False
    finally:
        cursor.close()


def check_book_exists(kyobo_id):
    """도서가 이미 books 테이블에 존재하는지 확인"""
    cursor = db.conn.cursor()
    try:
        query = "SELECT 1 FROM books WHERE kyobo_id = %s"
        cursor.execute(query, (kyobo_id,))
        return bool(cursor.fetchone())
    finally:
        cursor.close()


def process_single_book(kyobo_id, scraper):
    """단일 도서 처리"""
    book_url = f"https://product.kyobobook.co.kr/detail/{kyobo_id}"
    print(f"Processing book with kyobo_id: {kyobo_id} ({book_url})")

    try:
        book_details = scraper.scrape(book_url)

        if book_details:
            book_details['kyobo_id'] = kyobo_id

            # 기본 책 정보가 있는지 확인
            book_exists = check_book_exists(kyobo_id)

            if book_exists:
                # 책이 이미 존재하면 업데이트
                if db.update_book(book_details):
                    print(f"✅ 교보문고 상세 정보 업데이트 완료: {kyobo_id}")
                    mark_book_as_scraped(kyobo_id, True)
                    return True
                else:
                    print(f"❌ 교보문고 상세 정보 업데이트 실패: {kyobo_id}")
                    mark_book_as_scraped(kyobo_id, False)
                    return False
            else:
                # 책이 존재하지 않으면 기본 정보만 삽입
                isbn = book_details.get('isbn', '')
                title = book_details.get('title', '')

                book_data = {
                    'kyobo_id': kyobo_id,
                    'isbn': isbn,
                    'title': title
                }

                if db.insert_book(book_data) and db.update_book(book_details):
                    print(f"✅ 새 책 정보 추가 및 업데이트 완료: {kyobo_id}")
                    mark_book_as_scraped(kyobo_id, True)
                    return True
                else:
                    print(f"❌ 책 정보 추가 실패: {kyobo_id}")
                    mark_book_as_scraped(kyobo_id, False)
                    return False
        else:
            print(f"❌ 교보문고 상세 정보 스크래핑 실패: {kyobo_id}")
            mark_book_as_scraped(kyobo_id, False)
            return False

    except Exception as e:
        print(f"❌ 도서 처리 중 오류 발생: {kyobo_id}: {e}")
        mark_book_as_scraped(kyobo_id, False)
        return False


def save_state(processed_ids, successful_ids, failed_ids):
    """현재 진행 상태 저장"""
    state = {
        'processed_count': len(processed_ids),
        'successful_count': len(successful_ids),
        'failed_count': len(failed_ids),
        'last_processed_id': processed_ids[-1] if processed_ids else None,
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('scraper_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"Saved state: Processed {len(processed_ids)} books, Success: {len(successful_ids)}, Failed: {len(failed_ids)}")


def process_books(batch_size=50, max_books=None):
    """스크래핑 되지 않은 도서 배치 처리"""
    scraper = KyoboBookScraper()
    processed_ids = []
    successful_ids = []
    failed_ids = []
    total_processed = 0

    try:
        while True:
            if max_books and total_processed >= max_books:
                print(f"Reached maximum number of books to process: {max_books}")
                break

            current_batch_size = min(batch_size, max_books - total_processed if max_books else batch_size)
            unscraped_books = get_unscraped_books(current_batch_size)

            if not unscraped_books:
                print("No more unscraped books found.")
                break

            print(f"\n{'=' * 50}")
            print(f"Processing batch of {len(unscraped_books)} books...")
            print(f"{'=' * 50}\n")

            for i, kyobo_id in enumerate(unscraped_books, 1):
                try:
                    print(f"\n{'*' * 30}")
                    print(f"Processing book {i} of {len(unscraped_books)} in current batch")
                    print(f"{'*' * 30}")

                    success = process_single_book(kyobo_id, scraper)

                    processed_ids.append(kyobo_id)
                    if success:
                        successful_ids.append(kyobo_id)
                    else:
                        failed_ids.append(kyobo_id)

                    if i % 10 == 0 or i == len(unscraped_books):
                        save_state(processed_ids, successful_ids, failed_ids)

                    time.sleep(2)  # 요청 간 딜레이

                except Exception as e:
                    print(f"Error processing book {kyobo_id}: {e}")
                    failed_ids.append(kyobo_id)
                    save_state(processed_ids, successful_ids, failed_ids)

            total_processed += len(unscraped_books)
            print(f"\nCompleted processing batch. Total books processed so far: {total_processed}")

            if len(unscraped_books) < batch_size:
                print("No more books to process.")
                break

            print("Waiting before fetching next batch...")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Current state has been saved.")
        save_state(processed_ids, successful_ids, failed_ids)
    except Exception as e:
        print(f"\nError occurred: {e}. Current state has been saved.")
        save_state(processed_ids, successful_ids, failed_ids)

    return processed_ids, successful_ids, failed_ids


def main():
    # 커맨드 라인 인자 처리
    import argparse
    parser = argparse.ArgumentParser(description='교보문고 도서 스크래핑')
    parser.add_argument('--batch', type=int, default=50, help='한 번에 처리할 도서 수')
    parser.add_argument('--max', type=int, default=None, help='최대 처리할 도서 수')
    args = parser.parse_args()

    print(f"\n{'#' * 70}")
    print(f"Starting to process books from book_scraping table")
    print(f"Batch size: {args.batch}, Max books: {args.max or 'No limit'}")
    print(f"{'#' * 70}\n")

    processed_ids, successful_ids, failed_ids = process_books(args.batch, args.max)

    print(f"\nScraping completed!")
    print(f"Total processed: {len(processed_ids)}")
    print(f"Successful: {len(successful_ids)}")
    print(f"Failed: {len(failed_ids)}")

    # 실패한 ID들 기록
    if failed_ids:
        with open('failed_kyobo_ids.json', 'w', encoding='utf-8') as f:
            json.dump(failed_ids, f, ensure_ascii=False, indent=2)
        print(f"Failed IDs saved to failed_kyobo_ids.json")


if __name__ == "__main__":
    main()
