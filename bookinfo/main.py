#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from scraper import KyoboBookScraper


def print_result(book_info):
    """결과를 보기 좋게 출력하는 함수"""
    print("\n--- 최종 추출 결과 ---")
    for key, value in book_info.items():
        if value:
            if key == 'category':
                # JSON 문자열을 파싱하여 보기 좋게 출력
                try:
                    parsed_data = json.loads(value)
                    print(f"{key}:")
                    if isinstance(parsed_data, list):
                        for i, path in enumerate(parsed_data, 1):
                            print(f"  경로 {i}: {' > '.join(path)}")
                    elif isinstance(parsed_data, dict):
                        for grade_term, content in parsed_data.items():
                            print(f"  {grade_term}: {content}")
                    else:
                        print(f"  {parsed_data}")
                except:
                    print(f"{key}: {value}")
            elif key == 'description':
                # description은 줄바꿈을 유지하면서 일부만 출력
                preview = value[:100] + '...' if len(value) > 100 else value
                preview = preview.replace('\n', '\\n')
                print(f"{key}: {preview}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: 추출 실패")


def main():
    """메인 함수"""
    url = "https://product.kyobobook.co.kr/detail/S000208719345"

    print("도서 정보 스크래핑 시작")
    scraper = KyoboBookScraper()
    book_info = scraper.scrape(url)

    print_result(book_info)


if __name__ == "__main__":
    main()
