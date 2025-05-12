# naver_book_api.py

import requests
import json
import time
from urllib.parse import quote


class NaverBookAPI:
    def __init__(self, client_id, client_secret):
        """네이버 Open API 인증 정보를 초기화합니다."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search/book_adv"
        self.headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
            "Content-Type": "plain/text"
        }

    def search_by_isbn(self, isbn):
        """ISBN으로 도서 정보를 검색합니다."""
        # ISBN에서 하이픈 제거
        isbn = isbn.replace('-', '')

        # 쿼리 파라미터 설정
        params = {
            'd_isbn': isbn
        }

        try:
            # API 요청
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params
            )

            # 응답 코드 확인
            if response.status_code == 200:
                # JSON 응답 파싱
                result = response.json()

                # 검색 결과가 있는지 확인
                if result['total'] > 0 and len(result['items']) > 0:
                    return result['items'][0]  # 첫 번째 결과 반환
                else:
                    print(f"ISBN {isbn}에 대한 검색 결과가 없습니다.")
                    return None
            else:
                print(f"API 요청 실패: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"API 요청 중 오류 발생: {e}")
            return None

    def search_multiple_isbns(self, isbn_list):
        """여러 ISBN에 대한 도서 정보를 검색합니다."""
        results = {}

        for isbn in isbn_list:
            print(f"ISBN {isbn} 검색 중...")
            book_info = self.search_by_isbn(isbn)

            if book_info:
                results[isbn] = book_info
                print(f"ISBN {isbn} 검색 완료: {book_info['title']}")
            else:
                results[isbn] = None
                print(f"ISBN {isbn} 검색 결과 없음")

            # API 호출 간 딜레이 (초당 10회 이하로 제한됨)
            time.sleep(0.1)

        return results

    def merge_with_db_data(self, db_book, naver_book):
        """DB 데이터와 네이버 API 데이터를 병합합니다."""
        if not naver_book:
            return db_book

        merged_book = db_book.copy()

        # 네이버 API의 필드를 DB 필드에 매핑
        field_mapping = {
            'title': 'title',
            'subtitle': 'subtitle',  # 네이버 API는 부제목을 따로 제공하지 않을 수 있음
            'author': 'author',
            'publisher': 'publisher',
            'pubdate': 'publication_date',  # 형식 변환 필요 (YYYYMMDD -> YYYY-MM-DD)
            'description': 'description',
            'image': 'cover_image_url',
            'link': 'naver_link',  # 추가 필드
            'isbn': 'isbn'
        }

        # 필드 병합
        for naver_field, db_field in field_mapping.items():
            if naver_field in naver_book and naver_book[naver_field]:
                # 출판일 형식 변환
                if naver_field == 'pubdate' and len(naver_book[naver_field]) == 8:
                    year = naver_book[naver_field][:4]
                    month = naver_book[naver_field][4:6]
                    day = naver_book[naver_field][6:8]
                    merged_book[db_field] = f"{year}-{month}-{day}"
                # 그 외 필드는 그대로 복사
                elif naver_book[naver_field]:
                    # HTML 태그 제거
                    if naver_field in ['title', 'description', 'author', 'publisher']:
                        merged_book[db_field] = self._remove_html_tags(naver_book[naver_field])
                    else:
                        merged_book[db_field] = naver_book[naver_field]

        # 네이버 전용 필드 추가
        merged_book['naver_link'] = naver_book.get('link', '')
        merged_book['naver_discount'] = naver_book.get('discount', '')
        merged_book['naver_price'] = naver_book.get('price', '')

        # 카테고리 정보가 없으면 추가
        if not merged_book.get('category') and 'categoryName' in naver_book and naver_book['categoryName']:
            # 카테고리 이름을 배열로 변환
            categories = naver_book['categoryName'].split('>')
            categories = [cat.strip() for cat in categories]
            merged_book['category'] = categories

        return merged_book

    def _remove_html_tags(self, text):
        """HTML 태그를 제거합니다."""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)


# 사용 예시:
if __name__ == "__main__":
    # 네이버 Open API 클라이언트 초기화
    naver_api = NaverBookAPI(
        client_id="Ify9yiAgxNVLZrnrF7pV",
        client_secret="Svz3ireH_a"
    )

    # 단일 ISBN 검색 테스트
    isbn = "9791168417786"
    book_info = naver_api.search_by_isbn(isbn)

    if book_info:
        print(f"\n[{isbn}] 검색 결과:")
        print(f"제목: {book_info['title']}")
        print(f"저자: {book_info['author']}")
        print(f"출판사: {book_info['publisher']}")
        print(f"출판일: {book_info['pubdate']}")
        print(f"설명: {book_info['description'][:100]}...")
        print(f"이미지: {book_info['image']}")
        print(f"링크: {book_info['link']}")
    else:
        print(f"\n[{isbn}] 검색 결과가 없습니다.")

