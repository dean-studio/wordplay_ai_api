import requests
import json
import time
import os
import sys
from pathlib import Path

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


def fetch_kyobo_bestsellers(page=1, per_page=50, ymw='202407'):
    url = f"https://store.kyobobook.co.kr/api/gw/best/best-seller/online?page={page}&per=50&period=003&dsplDvsnCode=001&ymw={ymw}&dsplTrgtDvsnCode=004&saleCmdtClstCode=42"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from page {page}: {e}")
        return None


def process_single_book(book, scraper):
    processed_book = {
        'kyobo_id': book.get('saleCmdtid', ''),
        'isbn': book.get('cmdtCode', ''),
        'title': book.get('cmdtName', '')
    }

    book_url = f"https://product.kyobobook.co.kr/detail/{processed_book['kyobo_id']}"
    print(f"Processing book: {processed_book['title']} ({book_url})")

    try:
        book_data = {
            'kyobo_id': processed_book['kyobo_id'],
            'isbn': processed_book['isbn'],
            'title': processed_book['title']
        }

        insert_result = db.insert_book(book_data)
        if insert_result:
            print(f"✅ 기본 정보 저장 완료: {processed_book['title']}")

            isbn_list = [processed_book['isbn']]
            results = naver_api.search_multiple_isbns(isbn_list)

            update_success = False
            for isbn, info in results.items():
                if info:
                    book_data = {
                        'author': info['author'],
                        'publisher': info['publisher'],
                        'publication_date': info['pubdate'],
                        'kyobo_id': processed_book['kyobo_id']
                    }

                    update_success = db.update_book_info(book_data)
                    if update_success:
                        print(f"✅ 네이버 정보 업데이트 완료: {processed_book['title']}")
                    else:
                        print(f"❌ 네이버 정보 업데이트 실패: {processed_book['title']}")
                else:
                    print(f"ISBN {isbn}: 네이버 검색 결과 없음")

            if update_success or not results.get(processed_book['isbn']):
                book_details = scraper.scrape(book_url)

                if book_details:
                    book_details['kyobo_id'] = processed_book['kyobo_id']

                    if db.update_book(book_details):
                        print(f"✅ 교보문고 상세 정보 업데이트 완료: {processed_book['title']}")
                    else:
                        print(f"❌ 교보문고 상세 정보 업데이트 실패: {processed_book['title']}")

                    processed_book.update(book_details)
                else:
                    print(f"❌ 교보문고 상세 정보 스크래핑 실패: {processed_book['title']}")
        else:
            print(f"ℹ️ 이미 데이터베이스에 존재하는 도서: {processed_book['title']}")

        print(f"✅ 처리 완료: {processed_book['title']}")

    except Exception as e:
        print(f"❌ 도서 처리 중 오류 발생: {processed_book['title']}: {e}")

    time.sleep(2)

    return processed_book


def save_state(current_page, current_book_index, processed_data, current_ymw):
    state = {
        'current_page': current_page,
        'current_book_index': current_book_index,
        'current_ymw': current_ymw,
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('crawler_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    with open(f'kyobo_bestsellers_partial_{current_ymw}.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Saved state: YMW {current_ymw}, Page {current_page}, Book {current_book_index}")


def load_state():
    if os.path.exists('crawler_state.json'):
        try:
            with open('crawler_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)

            current_ymw = state.get('current_ymw', '202407')

            if os.path.exists(f'kyobo_bestsellers_partial_{current_ymw}.json'):
                with open(f'kyobo_bestsellers_partial_{current_ymw}.json', 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
            else:
                processed_data = []

            print(
                f"Loaded state: YMW {current_ymw}, Page {state['current_page']}, Book {state['current_book_index']}, Last updated: {state['last_updated']}")
            return state['current_page'], state['current_book_index'], processed_data, current_ymw
        except Exception as e:
            print(f"Error loading state: {e}")

    return 1, 0, [], '202407'


def process_all_pages(max_pages=5, resume=True, ymw='202407'):
    scraper = KyoboBookScraper()

    if resume:
        current_page, current_book_index, all_processed_data, current_ymw = load_state()
        if current_ymw != ymw:
            print(f"Starting new YMW period: {ymw}")
            current_page, current_book_index, all_processed_data = 1, 0, []
    else:
        current_page, current_book_index, all_processed_data = 1, 0, []
        current_ymw = ymw

    try:
        for page in range(current_page, max_pages + 1):
            print(f"\n{'=' * 50}")
            print(f"Processing YMW {ymw}, page {page}...")
            print(f"{'=' * 50}\n")

            raw_data = fetch_kyobo_bestsellers(page, 50, ymw)

            if not raw_data or 'data' not in raw_data or 'bestSeller' not in raw_data['data']:
                print(f"No valid data found on page {page}. Stopping.")
                save_state(page + 1, 0, all_processed_data, ymw)
                break

            books = raw_data['data']['bestSeller']
            if not books:
                print(f"No books found on page {page}. Stopping.")
                save_state(page + 1, 0, all_processed_data, ymw)
                break

            print(f"Found {len(books)} books on page {page}")

            start_book_index = current_book_index if page == current_page else 0

            for i, book in enumerate(books[start_book_index:], start_book_index + 1):
                try:
                    print(f"\n{'*' * 30}")
                    print(f"Processing book {i} of {len(books)} on page {page}")
                    print(f"{'*' * 30}")

                    processed_book = process_single_book(book, scraper)
                    all_processed_data.append(processed_book)

                    save_state(page, i, all_processed_data, ymw)

                except Exception as e:
                    print(f"Error processing book {i} on page {page}: {e}")
                    save_state(page, i, all_processed_data, ymw)
                    raise

            current_book_index = 0
            save_state(page + 1, current_book_index, all_processed_data, ymw)

            print(f"\nCompleted processing page {page}. Total books processed so far: {len(all_processed_data)}")

            if page < max_pages:
                print("Waiting before fetching next page...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Current state has been saved.")
        return all_processed_data
    except Exception as e:
        print(f"\nError occurred: {e}. Current state has been saved.")
        return all_processed_data

    with open(f'kyobo_bestsellers_complete_{ymw}.json', 'w', encoding='utf-8') as f:
        json.dump(all_processed_data, f, ensure_ascii=False, indent=2)
    print(f"All data for YMW {ymw} saved to kyobo_bestsellers_complete_{ymw}.json")

    return all_processed_data


def main():
    max_pages = 50
    ymw_periods = ['202503', '202504']

    for ymw in ymw_periods:
        print(f"\n{'#' * 70}")
        print(f"Starting to process YMW period: {ymw}")
        print(f"{'#' * 70}\n")

        resume = False
        if os.path.exists('crawler_state.json'):
            with open('crawler_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)
                current_ymw = state.get('current_ymw', '202407')

            if current_ymw == ymw:
                choice = input(f"Previous crawling state found for YMW {ymw}. Resume? (y/n): ").lower()
                resume = choice == 'y' or choice == ''

        processed_data = process_all_pages(max_pages, resume, ymw)

        if processed_data:
            print(f"\nTotal books processed for YMW {ymw}: {len(processed_data)}")
        else:
            print(f"No data was processed for YMW {ymw}.")

        if os.path.exists('crawler_state.json'):
            os.remove('crawler_state.json')
        if os.path.exists(f'kyobo_bestsellers_partial_{ymw}.json'):
            os.remove(f'kyobo_bestsellers_partial_{ymw}.json')

        print(f"Completed processing YMW period: {ymw}")
        time.sleep(10)  # 다음 ymw 처리 전 10초 대기


if __name__ == "__main__":
    main()