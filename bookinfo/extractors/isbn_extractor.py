#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from utils.html_parser import find_context


def extract_isbn(page_source):
    """ISBN 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        str: 추출된 ISBN 또는 None
    """
    # ISBN 패턴 매칭
    isbn_pattern = r'<th[^>]*>ISBN</th>\s*<td[^>]*>(\d+)</td>'
    isbn_match = re.search(isbn_pattern, page_source)

    if isbn_match:
        isbn = isbn_match.group(1)
        print(f"ISBN 추출 성공: {isbn}")
        return isbn

    print("ISBN 패턴 매치 실패, 대체 패턴 시도")

    # 대체 패턴
    alt_isbn_pattern = r'ISBN[^<>]*?(\d{13})'
    alt_isbn_match = re.search(alt_isbn_pattern, page_source)

    if alt_isbn_match:
        isbn = alt_isbn_match.group(1)
        print(f"ISBN 추출 성공 (대체 패턴): {isbn}")
        return isbn

    # 컨텍스트 출력 (디버깅용)
    isbn_context = find_context(page_source, 'ISBN')
    if isbn_context:
        print(f"ISBN 관련 HTML 부분:\n{isbn_context}")

    print("ISBN 추출 실패")
    return None