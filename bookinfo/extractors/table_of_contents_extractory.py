#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
from utils.html_parser import clean_html, find_context


def extract_table_of_contents(page_source):
    """목차 정보 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        str: 목차 정보 (JSON 문자열) 또는 None
    """
    table_of_contents = None

    # 목차 영역 추출
    toc_pattern = r'<div\s+class="product_detail_area\s+book_contents".*?>.*?</div>\s*</div>\s*</div>\s*</div>'
    toc_match = re.search(toc_pattern, page_source, re.DOTALL)

    if toc_match:
        toc_html = toc_match.group(0)
        print("목차 영역 HTML 발견")

        # book_contents_item 내의 텍스트 추출
        item_pattern = r'<li\s+class="book_contents_item">(.*?)</li>'
        item_match = re.search(item_pattern, toc_html, re.DOTALL)

        if item_match:
            item_html = item_match.group(1)
            print(f"목차 항목 HTML: {item_html[:100]}...")

            # <br> 태그로 분할
            # HTML 태그를 개행 문자로 변환하기 전에 <br> 태그를 특별한 마커로 대체
            item_html_marked = re.sub(r'<br\s*/?>', '###BR###', item_html)

            # 다른 HTML 태그 제거
            clean_item = re.sub(r'<[^>]*>', '', item_html_marked)

            # 마커를 다시 개행 문자로 변환하여 항목 분할
            toc_items = clean_item.split('###BR###')

            # 빈 항목 제거 및 공백 정리
            toc_items = [item.strip() for item in toc_items if item.strip()]

            # JSON으로 변환
            table_of_contents = json.dumps(toc_items, ensure_ascii=False)
            print(f"목차 추출 성공: {len(toc_items)}개 항목")
            print("목차 항목:")
            for i, item in enumerate(toc_items, 1):
                print(f"  {i}. {item}")
        else:
            print("목차 항목(book_contents_item)을 찾지 못했습니다.")

            # 대체 방법: 컨텐츠 영역 전체 텍스트에서 추출 시도
            contents_text = re.sub(r'<[^>]*>', ' ', toc_html)
            contents_text = re.sub(r'\s+', ' ', contents_text).strip()

            if "프롤로그" in contents_text and "사건" in contents_text:
                # 패턴 기반 추출
                toc_items = re.findall(r'(프롤로그[^:]*:|[^:]*사건[^:]*:)[^:]+', contents_text)

                if toc_items:
                    # 항목 정리
                    toc_items = [item.strip() for item in toc_items if item.strip()]

                    # JSON으로 변환
                    table_of_contents = json.dumps(toc_items, ensure_ascii=False)
                    print(f"목차 대체 추출 성공: {len(toc_items)}개 항목")
    else:
        print("목차 영역(product_detail_area book_contents)을 찾지 못했습니다.")

        # 컨텍스트 출력 (디버깅용)
        contents_context = find_context(page_source, "book_contents")
        if contents_context:
            print(f"목차 관련 HTML 부분:\n{contents_context}")

        # 대체 패턴 시도
        alt_toc_pattern = r'<div[^>]*>.*?목차.*?<ul[^>]*>(.*?)</ul>'
        alt_toc_match = re.search(alt_toc_pattern, page_source, re.DOTALL)

        if alt_toc_match:
            toc_html = alt_toc_match.group(1)

            # li 태그 내용 추출
            li_items = re.findall(r'<li[^>]*>(.*?)</li>', toc_html, re.DOTALL)

            if li_items:
                # HTML 태그 제거 및 정리
                toc_items = [clean_html(item) for item in li_items]
                toc_items = [item for item in toc_items if item.strip()]

                # JSON으로 변환
                table_of_contents = json.dumps(toc_items, ensure_ascii=False)
                print(f"목차 대체 패턴 추출 성공: {len(toc_items)}개 항목")


    return table_of_contents
