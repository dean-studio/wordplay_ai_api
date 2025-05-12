#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from selenium.common.exceptions import TimeoutException, WebDriverException

from bookinfo.extractors.publisher_review_extractory import extract_publisher_review
from bookinfo.extractors.table_of_contents_extractory import extract_table_of_contents
from utils.browser import setup_browser
from extractors.isbn_extractor import extract_isbn
from extractors.title_extractor import extract_title
from extractors.subtitle_extractor import extract_subtitle
from extractors.image_extractor import extract_cover_image, extract_description_image
from extractors.description_extractor import extract_description_and_info_texts
from extractors.category_extractor import extract_category


class KyoboBookScraper:
    """교보문고 도서 정보 스크래퍼 클래스"""

    def __init__(self, max_retries=3):
        """초기화 함수

        Args:
            max_retries (int): 스크래핑 실패 시 재시도 횟수
        """
        self.max_retries = max_retries

    def init_book_info(self):
        """도서 정보 초기화"""
        return {
            'isbn': None,
            'title': None,
            'subtitle': None,
            'cover_image_url': None,
            'description_image_url': None,
            'description': None,
            'curriculum_connection': None,
            'category': None
        }

    def extract_book_info(self, driver):
        """페이지에서 도서 정보 추출

        Args:
            driver: Selenium WebDriver 인스턴스

        Returns:
            dict: 추출된 도서 정보
        """
        book_info = self.init_book_info()
        page_source = driver.page_source
        print(f"페이지 소스 길이: {len(page_source)} 바이트")

        # 각 정보 추출
        book_info['isbn'] = extract_isbn(page_source)
        book_info['title'] = extract_title(page_source)
        book_info['subtitle'] = extract_subtitle(page_source)
        book_info['cover_image_url'] = extract_cover_image(page_source)
        book_info['description_image_url'] = extract_description_image(page_source)

        # 설명 및 교과 연계 정보 추출
        description, info_texts_json = extract_description_and_info_texts(page_source)
        book_info['description'] = description
        book_info['info_texts_json'] = info_texts_json

        # 카테고리 추출
        book_info['category'] = extract_category(page_source)

        book_info['table_of_contents'] = extract_table_of_contents(page_source)
        book_info['publisher_review'] = extract_publisher_review(page_source)
        return book_info

    def scrape(self, url):
        """URL에서 도서 정보 스크래핑

        Args:
            url (str): 스크래핑할 도서 페이지 URL

        Returns:
            dict: 추출된 도서 정보
        """
        for retry in range(self.max_retries):
            try:
                print(f"시도 {retry + 1}/{self.max_retries}...")

                # 브라우저 설정
                driver = setup_browser()
                book_info = self.init_book_info()

                try:
                    # 페이지 로딩
                    print("페이지 로딩 시작...")
                    start_time = time.time()
                    driver.get(url)

                    # 로딩 시간 측정 및 출력
                    load_time = time.time() - start_time
                    print(f"페이지 로딩 완료: {load_time:.2f}초")

                    # 도서 정보 추출
                    print("정보 추출 시작...")
                    book_info = self.extract_book_info(driver)

                    # 정보 추출 요약
                    print("\n--- 정보 추출 결과 요약 ---")
                    for key, value in book_info.items():
                        if value:
                            if key in ['description', 'category']:
                                print(f"{key}: [데이터 있음, 길이: {len(value)} 바이트]")
                            else:
                                print(f"{key}: {value}")
                        else:
                            print(f"{key}: 추출 실패")

                    # 주요 정보가 추출되었는지 확인
                    required_info = ['isbn', 'title', 'cover_image_url']
                    all_required_extracted = all(book_info[key] is not None for key in required_info)

                    if all_required_extracted:
                        print("필수 정보 추출 성공!")
                        return book_info

                    # 일부 정보만 추출된 경우
                    print("일부 정보만 추출됨. 추출된 정보 반환.")
                    return book_info

                except TimeoutException as te:
                    print(f"페이지 로딩 타임아웃: {te}")
                    if retry == self.max_retries - 1:  # 마지막 시도에서만 부분 정보 반환
                        return book_info
                except Exception as e:
                    print(f"스크래핑 중 오류 발생: {e}")
                    if retry == self.max_retries - 1:  # 마지막 시도에서만 부분 정보 반환
                        return book_info
                finally:
                    end_time = time.time()
                    total_time = end_time - start_time
                    print(f"총 실행 시간: {total_time:.2f}초")
                    driver.quit()

            except WebDriverException as wde:
                print(f"WebDriver 오류: {wde}")
                if retry == self.max_retries - 1:
                    return book_info
            except Exception as e:
                print(f"예상치 못한 오류: {e}")
                if retry == self.max_retries - 1:
                    return book_info

        return self.init_book_info()  # 모든 시도 실패 시 빈 정보 반환