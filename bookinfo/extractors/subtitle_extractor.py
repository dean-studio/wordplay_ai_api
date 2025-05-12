#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from utils.html_parser import find_context, clean_html


def extract_subtitle(page_source):
    """부제목 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        str: 추출된 부제목 또는 None
    """
    # 여러 부제목 패턴 시도
    subtitle_patterns = [
        r'<div[^>]*class="auto_overflow_inner"[^>]*>.*?<span[^>]*class="prod_desc"[^>]*>(.*?)</span>',
        r'<span[^>]*class="prod_desc"[^>]*>(.*?)</span>',
        r'<div[^>]*id="subTitleArea"[^>]*>.*?<span[^>]*class="prod_desc"[^>]*>(.*?)</span>'
    ]

    for i, pattern in enumerate(subtitle_patterns):
        subtitle_match = re.search(pattern, page_source, re.DOTALL)

        if subtitle_match:
            subtitle_text = subtitle_match.group(1).strip()
            print(f"부제목 패턴 {i + 1} 매치 성공: {subtitle_text}")

            # '|' 문자가 있으면 그 앞부분만 가져오기
            if '|' in subtitle_text:
                subtitle = subtitle_text.split('|')[0].strip()
                print(f"부제목 추출 성공 (| 이후 제거): {subtitle}")
            else:
                subtitle = subtitle_text
                print(f"부제목 추출 성공: {subtitle}")

            # HTML 태그 제거
            subtitle = clean_html(subtitle)
            return subtitle

    # 모든 패턴 실패 시 컨텍스트 출력 (디버깅용)
    prod_desc_context = find_context(page_source, 'prod_desc')
    if prod_desc_context:
        print(f"부제목 관련 HTML 부분:\n{prod_desc_context}")

    print("부제목 추출 실패")
    return None