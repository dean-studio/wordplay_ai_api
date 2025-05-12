# db_manager.py

import mysql.connector
from mysql.connector import Error
import json
import re
from datetime import datetime


class DatabaseManager:
    def __init__(self, host, user, password, db):
        """데이터베이스 연결 정보를 초기화합니다."""
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.connection = None

    def connect(self):
        """데이터베이스에 연결합니다."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.db
            )
            if self.connection.is_connected():
                print("MySQL 데이터베이스에 연결되었습니다.")
                return True
        except Error as e:
            print(f"MySQL 연결 오류: {e}")
            return False

    def close(self):
        """데이터베이스 연결을 종료합니다."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL 연결이 종료되었습니다.")

    def execute_query(self, query, params=None):
        """SQL 쿼리를 실행합니다."""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False

        cursor = None
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            self.connection.commit()
            return True
        except Error as e:
            print(f"쿼리 실행 오류: {e}")
            return False
        finally:
            if cursor:
                cursor.close()

    def fetch_all(self, query, params=None):
        """SELECT 쿼리를 실행하고 모든 결과를 반환합니다."""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None

        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return cursor.fetchall()
        except Error as e:
            print(f"쿼리 실행 오류: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    def fetch_one(self, query, params=None):
        """SELECT 쿼리를 실행하고 첫 번째 결과를 반환합니다."""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None

        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return cursor.fetchone()
        except Error as e:
            print(f"쿼리 실행 오류: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    def insert_book(self, book_data):
        """책 정보를 데이터베이스에 삽입합니다. kyobo_id가 존재하지 않는 경우에만 삽입합니다."""
        # 필수 필드 확인
        kyobo_id = book_data.get('kyobo_id', '')
        if not kyobo_id:
            print("오류: kyobo_id는 필수 필드입니다.")
            return False

        # 이미 존재하는지 확인
        check_query = "SELECT id FROM kyobo_books WHERE kyobo_id = %s"
        existing_book = self.fetch_one(check_query, (kyobo_id,))

        # 이미 존재하면 삽입하지 않음
        if existing_book:
            print(f"도서 kyobo_id={kyobo_id}는 이미 데이터베이스에 존재합니다.")
            return True  # 이미 존재하므로 성공으로 간주

        # 존재하지 않으면 삽입
        insert_query = """
        INSERT INTO kyobo_books (
            kyobo_id, isbn, title
        ) VALUES (%s, %s, %s)
        """

        # 필요한 필드 추출
        isbn = book_data.get('isbn', '')
        title = book_data.get('title', '')

        # 매개변수 튜플 생성
        params = (kyobo_id, isbn, title)

        result = self.execute_query(insert_query, params)
        if result:
            print(f"도서 '{title}' (kyobo_id={kyobo_id}) 기본 정보 삽입 완료")
        else:
            print(f"도서 '{title}' (kyobo_id={kyobo_id}) 기본 정보 삽입 실패")

        return result

    def update_book_info(self, book_data):
        """책 정보를 업데이트합니다."""
        update_query = """
           UPDATE kyobo_books SET 
               author = %s,
               publication_date = %s,
               publisher = %s
           WHERE kyobo_id = %s
           """

        author = book_data.get('author', '')
        publication_date = book_data.get('publication_date', '')
        publisher = book_data.get('publisher', '')
        kyobo_id  = book_data.get('kyobo_id', '')

        params = (author, publication_date, publisher, kyobo_id)
        return self.execute_query(update_query, params)

    def update_book(self, book_data):
        """책 정보를 업데이트합니다. 제공된 book_details 구조에 맞게 조정됨."""
        # 필수 필드 확인
        if not book_data.get('kyobo_id'):
            print("오류: kyobo_id는 필수 필드입니다.")
            return False

        update_query = """
        UPDATE kyobo_books SET 
            subtitle = %s,
            cover_image_url = %s,
            category = %s,
            description_image_url = %s,
            description = %s,
            curriculum_connection = %s,
            table_of_contents = %s,
            publisher_review = %s,
            is_updated = 1
        WHERE kyobo_id = %s
        """

        # book_data에서 필요한 필드 추출
        subtitle = book_data.get('subtitle')
        cover_image_url = book_data.get('cover_image_url', '')

        # 카테고리 처리 - 이미 JSON 문자열이면 그대로, 아니면 변환
        category = book_data.get('category')
        if isinstance(category, (list, dict)) and not isinstance(category, str):
            category = json.dumps(category, ensure_ascii=False)

        description_image_url = book_data.get('description_image_url', '')
        description = book_data.get('description', '')
        curriculum_connection = book_data.get('curriculum_connection')

        # table_of_contents 처리 - JSON 문자열이면 그대로, 리스트면 문자열로 변환
        table_of_contents = book_data.get('table_of_contents')
        if isinstance(table_of_contents, str) and table_of_contents.startswith('['):
            # 이미 JSON 문자열 형태
            pass
        elif isinstance(table_of_contents, list):
            table_of_contents = json.dumps(table_of_contents, ensure_ascii=False)

        publisher_review = book_data.get('publisher_review', '')
        kyobo_id = book_data.get('kyobo_id')

        # 매개변수 튜플 생성
        params = (
            subtitle,
            cover_image_url,
            category,
            description_image_url,
            description,
            curriculum_connection,
            table_of_contents,
            publisher_review,
            kyobo_id
        )

        result = self.execute_query(update_query, params)
        if result:
            print(f"도서 정보 업데이트 성공: kyobo_id={kyobo_id}")
        else:
            print(f"도서 정보 업데이트 실패: kyobo_id={kyobo_id}")

        return result

    def insert_or_update_book(self, book_data):
        """책 정보를 삽입하거나 업데이트합니다."""
        if not book_data.get('kyobo_id'):
            print("오류: kyobo_id가 필요합니다.")
            return False

        # 책이 이미 존재하는지 확인
        check_query = "SELECT id FROM kyobo_books WHERE kyobo_id = %s"
        existing_book = self.fetch_one(check_query, (book_data.get('kyobo_id'),))

        if existing_book:
            return self.update_book(book_data)
        else:
            return self.insert_book(book_data)

    def get_book_by_id(self, kyobo_id):
        """kyobo_id로 책 정보를 검색합니다."""
        query = "SELECT * FROM kyobo_books WHERE kyobo_id = %s"
        return self.fetch_one(query, (kyobo_id,))

    def get_book_by_isbn(self, isbn):
        """ISBN으로 책 정보를 검색합니다."""
        query = "SELECT * FROM kyobo_books WHERE isbn = %s"
        return self.fetch_one(query, (isbn,))

    def get_books_by_title(self, title):
        """제목으로 책 정보를 검색합니다."""
        query = "SELECT * FROM kyobo_books WHERE title LIKE %s"
        return self.fetch_all(query, (f'%{title}%',))

    def get_all_books(self, limit=100, offset=0):
        """모든 책 정보를 가져옵니다."""
        query = "SELECT * FROM kyobo_books ORDER BY id DESC LIMIT %s OFFSET %s"
        return self.fetch_all(query, (limit, offset))

    def get_not_updated_books(self, limit=100):
        """아직 업데이트되지 않은 책 정보를 가져옵니다."""
        query = "SELECT * FROM kyobo_books WHERE is_updated = 0 ORDER BY id DESC LIMIT %s"
        return self.fetch_all(query, (limit,))

    def mark_book_as_updated(self, kyobo_id):
        """책을 업데이트 완료로 표시합니다."""
        query = "UPDATE kyobo_books SET is_updated = 1 WHERE kyobo_id = %s"
        return self.execute_query(query, (kyobo_id,))

    def _prepare_book_params(self, book_data):
        """책 데이터를 쿼리 파라미터로 변환합니다."""
        kyobo_id = book_data.get('kyobo_id', '')
        isbn = book_data.get('isbn', '')
        title = book_data.get('title', '')
        subtitle = book_data.get('subtitle', '')
        author = book_data.get('author', '')

        # 날짜 형식 변환
        publication_date = None
        if 'publication_date' in book_data and book_data['publication_date']:
            try:
                date_str = book_data['publication_date']
                # '2023년 5월 30일' 형식 변환
                if '년' in date_str and '월' in date_str:
                    date_parts = date_str.replace('년', '-').replace('월', '-').replace('일', '').split('-')
                    if len(date_parts) >= 3:
                        year, month, day = date_parts[:3]
                        month = month.strip().zfill(2)
                        day = day.strip().zfill(2)
                        publication_date = f"{year.strip()}-{month}-{day}"
                else:
                    # 이미 YYYY-MM-DD 형식인 경우
                    publication_date = date_str
            except Exception as e:
                print(f"날짜 형식 변환 오류: {e}")

        publisher = book_data.get('publisher', '')
        cover_image_url = book_data.get('cover_image_url', '')

        # 카테고리 JSON 변환
        category = None
        if 'category' in book_data and book_data['category']:
            category = json.dumps(book_data['category'], ensure_ascii=False)

        description_image_url = book_data.get('description_image_url', '')
        description = book_data.get('description', '')
        curriculum_connection = book_data.get('curriculum_connection', '')
        table_of_contents = book_data.get('table_of_contents', '')
        publisher_review = book_data.get('publisher_review', '')

        # 업데이트 상태
        is_updated = book_data.get('is_updated', 1)  # 기본값은 1 (업데이트됨)

        return (kyobo_id, isbn, title, subtitle, author, publication_date, publisher,
                cover_image_url, category, description_image_url, description,
                curriculum_connection, table_of_contents, publisher_review, is_updated)


# 사용 예시:
if __name__ == "__main__":
    # 데이터베이스 연결 테스트
    db = DatabaseManager(
        host='wordplayapi.mycafe24.com',
        user='wordplayapi',
        password='Hazbola2021!',
        db='wordplayapi'
    )

    if db.connect():
        print("연결 성공!")

        # 테스트 데이터
        test_book = {
            'kyobo_id': 'S000000001',
            'isbn': '9788956746425',
            'title': '테스트 책',
            'subtitle': '테스트 부제목',
            'author': '테스트 작가',
            'publisher': '테스트 출판사',
            'publication_date': '2023년 5월 30일',
            'cover_image_url': 'https://example.com/cover.jpg',
            'category': ['소설', '한국소설'],
            'description_image_url': 'https://example.com/desc.jpg',
            'description': '테스트 책 설명입니다.',
            'curriculum_connection': '교과 연계 정보입니다.',
            'table_of_contents': '목차 정보입니다.',
            'publisher_review': '출판사 리뷰입니다.',
            'is_updated': 1
        }

        # 삽입 또는 업데이트
        if db.insert_or_update_book(test_book):
            print("책 정보 저장 완료")

        # 조회 테스트
        book = db.get_book_by_id('S000000001')
        if book:
            print(f"검색된 책: {book['title']}")

        db.close()
    else:
        print("연결 실패")