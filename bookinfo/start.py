import requests
import json
import time
import os
import sys
from pathlib import Path

from db_manager import DatabaseManager
from naver_book_api import NaverBookAPI
# 상위 디렉토리에 있는 bookinfo 모듈을 import 하기 위한 설정
# sys.path.append(str(Path(__file__).parent.parent))
from main import KyoboBookScraper

# scraper = KyoboBookScraper()

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

def fetch_kyobo_bestsellers(page=1, per_page=50):
    # url = f"https://store.kyobobook.co.kr/api/gw/best/best-seller/online?page=2&per=50&period=003&dsplDvsnCode=001&ymw=202406&dsplTrgtDvsnCode=004&saleCmdtClstCode=42"
    url = f"https://store.kyobobook.co.kr/api/gw/best/best-seller/online?page={page}&per=50&period=003&dsplDvsnCode=001&ymw=202407&dsplTrgtDvsnCode=004&saleCmdtClstCode=42"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from page {page}: {e}")
        return None


def process_single_book(book, scraper):
    # 기본 정보 추출
    processed_book = {
        'kyobo_id': book.get('saleCmdtid', ''),
        'isbn': book.get('cmdtCode', ''),
        'title': book.get('cmdtName', '')
    }

    # 도서 상세 URL 생성
    book_url = f"https://product.kyobobook.co.kr/detail/{processed_book['kyobo_id']}"
    print(f"Processing book: {processed_book['title']} ({book_url})")

    try:
        # 기본 정보 저장
        book_data = {
            'kyobo_id': processed_book['kyobo_id'],
            'isbn': processed_book['isbn'],
            'title': processed_book['title']
        }

        insert_result = db.insert_book(book_data)
        if insert_result:
            print(f"✅ 기본 정보 저장 완료: {processed_book['title']}")

            # 네이버 API에서 추가 정보 가져오기
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

                    # 네이버 정보로 업데이트
                    update_success = db.update_book_info(book_data)
                    if update_success:
                        print(f"✅ 네이버 정보 업데이트 완료: {processed_book['title']}")
                    else:
                        print(f"❌ 네이버 정보 업데이트 실패: {processed_book['title']}")
                else:
                    print(f"ISBN {isbn}: 네이버 검색 결과 없음")

            # 네이버 업데이트가 성공하거나 결과가 없는 경우에만 스크래퍼 실행
            if update_success or not results.get(processed_book['isbn']):
                book_details = scraper.scrape(book_url)

                if book_details:
                    # book_details에 kyobo_id 추가 (업데이트에 필요)
                    book_details['kyobo_id'] = processed_book['kyobo_id']

                    # 스크래핑한 상세 정보로 다시 업데이트
                    if db.update_book(book_details):
                        print(f"✅ 교보문고 상세 정보 업데이트 완료: {processed_book['title']}")
                    else:
                        print(f"❌ 교보문고 상세 정보 업데이트 실패: {processed_book['title']}")

                    # processed_book 객체에 상세 정보 병합
                    processed_book.update(book_details)
                else:
                    print(f"❌ 교보문고 상세 정보 스크래핑 실패: {processed_book['title']}")
        else:
            print(f"ℹ️ 이미 데이터베이스에 존재하는 도서: {processed_book['title']}")

        print(f"✅ 처리 완료: {processed_book['title']}")

    except Exception as e:
        print(f"❌ 도서 처리 중 오류 발생: {processed_book['title']}: {e}")

    # 크롤링 간 간격을 두어 서버에 부담을 줄입니다
    time.sleep(2)

    return processed_book


def save_state(current_page, current_book_index, processed_data):
    state = {
        'current_page': current_page,
        'current_book_index': current_book_index,
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # 상태 파일 저장
    with open('crawler_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    # 지금까지 처리된 데이터 저장
    with open('kyobo_bestsellers_partial.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Saved state: Page {current_page}, Book {current_book_index}")


def load_state():
    if os.path.exists('crawler_state.json'):
        try:
            with open('crawler_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)

            # 이전에 저장된 데이터가 있는지 확인
            if os.path.exists('kyobo_bestsellers_partial.json'):
                with open('kyobo_bestsellers_partial.json', 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
            else:
                processed_data = []

            print(
                f"Loaded state: Page {state['current_page']}, Book {state['current_book_index']}, Last updated: {state['last_updated']}")
            return state['current_page'], state['current_book_index'], processed_data
        except Exception as e:
            print(f"Error loading state: {e}")

    # 상태 파일이 없거나 오류가 발생하면 처음부터 시작
    return 1, 0, []


def process_all_pages(max_pages=5, resume=True):
    # KyoboBookScraper 인스턴스 생성
    scraper = KyoboBookScraper()

    # 상태 로드 (이어서 시작할지 여부에 따라)
    if resume:
        current_page, current_book_index, all_processed_data = load_state()
    else:
        current_page, current_book_index, all_processed_data = 1, 0, []

    try:
        # 지정된 페이지부터 최대 페이지까지 처리
        for page in range(current_page, max_pages + 1):
            print(f"\n{'=' * 50}")
            print(f"Processing page {page}...")
            print(f"{'=' * 50}\n")

            raw_data = fetch_kyobo_bestsellers(page)

            if not raw_data or 'data' not in raw_data or 'bestSeller' not in raw_data['data']:
                print(f"No valid data found on page {page}. Stopping.")
                save_state(page + 1, 0, all_processed_data)  # 다음 페이지부터 시작하도록 저장
                break

            books = raw_data['data']['bestSeller']
            if not books:
                print(f"No books found on page {page}. Stopping.")
                save_state(page + 1, 0, all_processed_data)  # 다음 페이지부터 시작하도록 저장
                break

            print(f"Found {len(books)} books on page {page}")

            # 이 페이지에서 시작할 책 인덱스 결정
            start_book_index = current_book_index if page == current_page else 0

            # 각 책을 순차적으로 처리
            for i, book in enumerate(books[start_book_index:], start_book_index + 1):
                try:
                    print(f"\n{'*' * 30}")
                    print(f"Processing book {i} of {len(books)} on page {page}")
                    print(f"{'*' * 30}")

                    processed_book = process_single_book(book, scraper)
                    all_processed_data.append(processed_book)

                    # 매 책 처리 후 상태 저장
                    save_state(page, i, all_processed_data)

                except Exception as e:
                    print(f"Error processing book {i} on page {page}: {e}")
                    save_state(page, i, all_processed_data)  # 오류 발생 시 현재 상태 저장
                    raise  # 예외를 다시 발생시켜 처리 중단

            # 페이지 처리 완료 시 상태 저장
            current_book_index = 0  # 다음 페이지는 처음부터 시작
            save_state(page + 1, current_book_index, all_processed_data)

            # 페이지 처리 완료 메시지
            print(f"\nCompleted processing page {page}. Total books processed so far: {len(all_processed_data)}")

            # 페이지 간 간격
            if page < max_pages:
                print("Waiting before fetching next page...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Current state has been saved.")
        return all_processed_data
    except Exception as e:
        print(f"\nError occurred: {e}. Current state has been saved.")
        return all_processed_data

    # 모든 페이지 처리 완료 시 상태 파일 삭제
    if os.path.exists('crawler_state.json'):
        os.remove('crawler_state.json')
    if os.path.exists('kyobo_bestsellers_partial.json'):
        os.remove('kyobo_bestsellers_partial.json')

    return all_processed_data


def main():
    # 처리할 최대 페이지 수 설정
    max_pages = 50

    # 이어서 시작할지 여부 확인
    resume = False
    if os.path.exists('crawler_state.json'):
        choice = input("Previous crawling state found. Resume? (y/n): ").lower()
        resume = choice == 'y' or choice == ''  # 엔터키도 yes로 처리

    print(f"Starting to process up to {max_pages} pages of Kyobo bestsellers")
    processed_data = process_all_pages(max_pages, resume)

    if processed_data:
        print(f"\nTotal books processed across all pages: {len(processed_data)}")

        # 결과 저장
        with open('kyobo_bestsellers_complete.json', 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print("All data saved to kyobo_bestsellers_complete.json")
    else:
        print("No data was processed.")


if __name__ == "__main__":
    main()