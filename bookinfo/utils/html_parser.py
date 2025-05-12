#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re


def clean_html(html_text):
    """HTML 태그 제거 및 텍스트 정리

    Args:
        html_text (str): HTML 문자열

    Returns:
        str: 정리된 텍스트
    """
    # <br> 태그는 개행문자로 변환
    text = re.sub(r'<br\s*/?>', '\n', html_text)
    # 나머지 HTML 태그 제거
    text = re.sub(r'<[^>]*>', '', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text


def find_context(page_source, search_term, context_size=100):
    """페이지 소스에서 검색어 주변 컨텍스트 추출

    Args:
        page_source (str): 페이지 소스
        search_term (str): 검색어
        context_size (int): 앞뒤로 추출할 문자 수

    Returns:
        str: 추출된 컨텍스트 또는 None
    """
    index = page_source.find(search_term)
    if index != -1:
        start = max(0, index - context_size)
        end = min(len(page_source), index + context_size)
        return page_source[start:end]
    return None