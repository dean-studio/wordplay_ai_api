#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
from utils.html_parser import clean_html, find_context


def extract_category(page_source):
    """카테고리 정보 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        str: 추출된 카테고리 정보 (JSON 문자열) 또는 None
    """
    # 카테고리 목록 찾기
    category_list_pattern = r'<ul\s+class="intro_category_list">(.*?)</ul>'
    category_list_match = re.search(category_list_pattern, page_source, re.DOTALL)

    if category_list_match:
        category_list_html = category_list_match.group(1)
        print("카테고리 목록 HTML 발견")

        # 카테고리 항목 파싱
        category_items = re.findall(r'<li\s+class="category_list_item">(.*?)</li>', category_list_html, re.DOTALL)
        print(f"카테고리 항목 {len(category_items)}개 발견")

        category_paths = []

        for idx, item in enumerate(category_items):
            # 각 항목에서 카테고리 링크 텍스트 추출
            links = re.findall(r'<a\s+href="[^"]*"\s+class="intro_category_link">(.*?)</a>', item)

            path = []
            for link_text in links:
                clean_text = clean_html(link_text)
                if clean_text:
                    path.append(clean_text)

            if path:
                category_paths.append(path)
                print(f"경로 {idx + 1} 추출: {' > '.join(path)}")

        if category_paths:
            category_json = json.dumps(category_paths, ensure_ascii=False)
            print(f"카테고리 파싱 성공: {len(category_paths)}개 경로")
            return category_json
    else:
        print("카테고리 목록 HTML을 찾지 못함, 대체 방법 시도")

        # 대체 방법: 가능한 다른 카테고리 구조 확인
        alt_category_pattern = r'<div[^>]*class="[^"]*category[^"]*"[^>]*>(.*?)</div>'
        alt_category_match = re.search(alt_category_pattern, page_source, re.DOTALL)

        if alt_category_match:
            category_text = clean_html(alt_category_match.group(1))
            if '>' in category_text:
                # '>' 기호로 구분된 카테고리 경로
                path = [part.strip() for part in category_text.split('>')]
                path = [p for p in path if p]  # 빈 항목 제거

                if path:
                    category_json = json.dumps([path], ensure_ascii=False)
                    print(f"카테고리 파싱 성공 (대체 방법): {' > '.join(path)}")
                    return category_json

    # 정보가 없을 경우 컨텍스트 출력 (디버깅용)
    category_context = find_context(page_source, 'category')
    if category_context:
        print(f"카테고리 관련 HTML 부분:\n{category_context}")

    print("카테고리 추출 실패")
    return None