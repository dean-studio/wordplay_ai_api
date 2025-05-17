import requests
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

from db_manager import DatabaseManager
from naver_book_api import NaverBookAPI
from main import KyoboBookScraper

naver_api = NaverBookAPI(
    client_id="Ify9yiAgxNVLZrnrF7pV",
    client_secret="Svz3ireH_a"
)

db = DatabaseManager(
    host='wordplayapi.mycafe24.com',
    user='wordplayapi',
    password='Hazbola2021!',
    db='wordplayapi'
)


def get_unscraped_books(limit=50):
    query = """
    SELECT kyobo_id 
    FROM book_scraping 
    WHERE is_scraped = FALSE 
    LIMIT %s
    """
    result = db.execute_query(query, (limit,))
    return [row[0] for row in result] if result else []


def mark_book_as_scraped(kyobo_id, success=True):
    query = """
    UPDATE book_scraping 
    SET is_scraped = %s, scraped_at = %s 
    WHERE kyobo_id = %s
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return db.execute_query(query, (success, current_time, kyobo_id))


def process_single_book(kyobo_id, scraper):
    book_url = f"https://product.kyobobook.co.kr/detail/{kyobo_id}"
    print(f"Processing book with kyobo_id: {kyobo_id} ({book_url})")

    try:
        book_details = scraper.scrape(book_url)

        if book_details:
            book_details['kyobo_id'] = kyobo_id

            # 기본 책 정보가 있는지 확인
            book_exists = db.execute_query(
                "SELECT 1 FROM books WHERE kyobo_id = %s",
                (kyobo_id,)
            )

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

                    # ISBN이 있으면 네이버 API로 추가 정보 가져오기
                    if isbn:
                        results = naver_api.search_multiple_isbns([isbn])

                        for isbn, info in results.items():
                            if info:
                                naver_data = {
                                    'author': info['author'],
                                    'publisher': info['publisher'],
                                    'publication_date': info['pubdate'],
                                    'kyobo_id': kyobo_id
                                }

                                if db.update_book_info(naver_data):
                                    print(f"✅ 네이버 정보 업데이트 완료: {kyobo_id}")

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
    except Exception as e:
        print(f"\nError occurred: {e}. Current state has been saved.")

    print(f"\nTotal books processed: {len(processed_ids)}")
    print(f"Successful: {len(successful_ids)}")
    print(f"Failed: {len(failed_ids)}")

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
