#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
from utils.html_parser import clean_html, find_context


def extract_publisher_review(page_source):
    """출판사 서평 정보 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        str: 출판사 서평 또는 None
    """
    publisher_review = None

    # 출판사 서평 영역 추출
    review_pattern = r'<div\s+class="product_detail_area\s+book_publish_review".*?>.*?<p\s+class="info_text">(.*?)</p>'
    review_match = re.search(review_pattern, page_source, re.DOTALL)

    if review_match:
        review_html = review_match.group(1)
        print("출판사 서평 HTML 발견")

        # HTML 태그 처리 (<br> 태그는 줄바꿈으로 변환)
        publisher_review = clean_html(review_html)
        print(f"출판사 서평 추출 성공: {len(publisher_review)} 바이트")
        print(f"서평 미리보기: {publisher_review[:150]}...")
    else:
        print("출판사 서평(book_publish_review)을 찾지 못했습니다.")

        # 대체 패턴 시도
        alt_review_pattern = r'<div[^>]*>.*?출판사\s*서평.*?<p[^>]*>(.*?)</p>'
        alt_review_match = re.search(alt_review_pattern, page_source, re.DOTALL)

        if alt_review_match:
            review_html = alt_review_match.group(1)
            publisher_review = clean_html(review_html)
            print(f"출판사 서평 대체 패턴 추출 성공: {len(publisher_review)} 바이트")
        else:
            # 컨텍스트 출력 (디버깅용)
            review_context = find_context(page_source, "book_publish_review")
            if review_context:
                print(f"출판사 서평 관련 HTML 부분:\n{review_context}")
            else:
                review_context = find_context(page_source, "출판사 서평")
                if review_context:
                    print(f"출판사 서평 텍스트 주변 HTML:\n{review_context}")

    return publisher_review
