#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from utils.html_parser import find_context


def extract_title(page_source):
    """제목 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        str: 추출된 제목 또는 None
    """
    # 제목 패턴 매칭
    title_pattern = r'<span[^>]*class="prod_title"[^>]*>(.*?)</span>'
    title_match = re.search(title_pattern, page_source)

    if title_match:
        title = title_match.group(1).strip()
        print(f"제목 추출 성공: {title}")
        return title

    print("제목 패턴 매치 실패, 대체 패턴 시도")

    # 대체 패턴 (h1 태그 내 제목)
    alt_title_pattern = r'<h1[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</h1>'
    alt_title_match = re.search(alt_title_pattern, page_source, re.DOTALL)

    if alt_title_match:
        # HTML 태그 제거
        title_with_tags = alt_title_match.group(1)
        title = re.sub(r'<[^>]*>', '', title_with_tags).strip()
        print(f"제목 추출 성공 (대체 패턴): {title}")
        return title

    # 컨텍스트 출력 (디버깅용)
    title_context = find_context(page_source, 'prod_title')
    if title_context:
        print(f"제목 관련 HTML 부분:\n{title_context}")

    print("제목 추출 실패")
    return None