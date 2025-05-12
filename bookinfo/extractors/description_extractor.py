#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
from utils.html_parser import clean_html, find_context


def extract_description_and_info_texts(page_source):
    """책 설명 및 info_texts JSON 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        tuple: (description, info_texts_json) - 추출된 설명, info_texts JSON
    """
    description = None
    info_texts_json = None  # 모든 info_text를 JSON으로 순서대로 저장할 변수

    # 1. intro_bottom 패턴 검색 - 여러 패턴 시도
    intro_bottom_patterns = [
        r'<div\s+class="intro_bottom">(.*?)</div>\s*</div>',  # 첫 번째 패턴
        r'<div\s+class="intro_bottom">(.*?)</div>',  # 두 번째 패턴 (단일 닫는 태그)
        r'<div[^>]*class="intro_bottom"[^>]*>(.*?)(?=<div\s+class=|</div>)'  # 세 번째 패턴 (더 유연한 매칭)
    ]

    intro_bottom_html = None
    matched_pattern = None

    for pattern in intro_bottom_patterns:
        intro_bottom_match = re.search(pattern, page_source, re.DOTALL)
        if intro_bottom_match:
            intro_bottom_html = intro_bottom_match.group(1)
            matched_pattern = pattern
            break

    if intro_bottom_html:
        print(f"intro_bottom HTML 발견 (패턴: {matched_pattern})")
        print(f"발견된 HTML 미리보기: {intro_bottom_html[:200]}...")

        # info_text 요소 개수 확인 (중요)
        info_text_count = len(re.findall(r'<div\s+class="info_text', intro_bottom_html))
        print(f"info_text 요소 수: {info_text_count}")

        # 모든 info_text 요소 추출 - 패턴 수정
        info_text_patterns = [
            r'<div[^>]*class="info_text[^"]*"[^>]*>(.*?)</div>',  # 기본 패턴
            r'<div\s+class="info_text">(.*?)</div>',  # 간단한 패턴
            r'<div[^>]*class=["\']info_text["\'][^>]*>(.*?)</div>'  # 따옴표 처리 강화
        ]

        info_texts = []
        matched_info_pattern = None

        for pattern in info_text_patterns:
            found_texts = re.findall(pattern, intro_bottom_html, re.DOTALL)
            if found_texts:
                info_texts = found_texts
                matched_info_pattern = pattern
                break

        print(f"추출된 info_text 요소 수: {len(info_texts)}")
        if matched_info_pattern:
            print(f"매칭된 info_text 패턴: {matched_info_pattern}")

        # 추출된 요소가 없으면 intro_bottom 자체에 텍스트가 있는지 확인
        if not info_texts:
            print("info_text 요소를 찾지 못함 - 대체 방법 시도")

            # 직접 HTML 검사
            if '<div class="info_text">' in intro_bottom_html:
                print("'<div class=\"info_text\">' 문자열 발견 - 직접 파싱 시도")

                # 시작 인덱스 찾기
                start_idx = intro_bottom_html.find('<div class="info_text">')

                if start_idx != -1:
                    # 시작 위치 다음부터 문자열 자르기
                    sub_html = intro_bottom_html[start_idx:]

                    # 닫는 태그 찾기
                    end_tag_idx = sub_html.find('</div>')

                    if end_tag_idx != -1:
                        # 시작 태그 길이 계산
                        start_tag_len = len('<div class="info_text">')

                        # 내용 추출
                        content = sub_html[start_tag_len:end_tag_idx]
                        info_texts = [content]
                        print(f"직접 파싱으로 info_text 내용 찾음: {content[:100]}...")

            # 그래도 못 찾으면 전체 내용 사용
            if not info_texts:
                print("직접 파싱으로도 찾지 못함 - intro_bottom 전체 내용 사용")
                # intro_bottom 내부 태그 제거 후 텍스트 추출
                clean_text = clean_html(intro_bottom_html)
                if clean_text.strip():
                    info_texts = [intro_bottom_html]  # 전체 HTML을 하나의 요소로 취급
                    print(f"intro_bottom에서 직접 텍스트 추출: {clean_text[:100]}...")

        if info_texts:
            # info_texts를 JSON으로 변환하기 위한 리스트
            info_texts_list = []

            # description: 모든 info_text 내용을 합침
            description_parts = []

            for i, info_text in enumerate(info_texts):
                # HTML 태그 제거 (단, <br>는 개행문자로 변환)
                clean_text = clean_html(info_text)
                print(f" [ clean_text {i + 1} ] {clean_text[:100]}...")

                # 정제된 텍스트를 info_texts_list에 추가
                info_texts_list.append(clean_text)

                if clean_text:
                    description_parts.append(clean_text)
                    print(f" [ clean_text in {i + 1} ] {clean_text[:100]}...")

            # 최종 description 생성
            if description_parts:
                description = '\n\n'.join(description_parts)
                print(f"\n설명(description) 추출 성공: {len(description)} 바이트")
                print("설명 처음 200자:")
                print(description[:200] + "..." if len(description) > 200 else description)

            # info_texts를 JSON으로 변환
            info_texts_json = json.dumps(info_texts_list, ensure_ascii=False)
            print(f"\ninfo_texts JSON 생성 성공: {len(info_texts_json)} 바이트")
            print("info_texts JSON 미리보기:")
            print(info_texts_json[:200] + "..." if len(info_texts_json) > 200 else info_texts_json)
    else:
        print("intro_bottom을 찾지 못했습니다.")

        # 직접 info_text 클래스 요소 찾기 시도
        info_texts = re.findall(r'<div[^>]*class="info_text[^"]*"[^>]*>(.*?)</div>', page_source, re.DOTALL)

        if info_texts:
            print(f"페이지에서 직접 info_text 요소 {len(info_texts)}개 발견")

            # info_texts를 JSON으로 변환하기 위한 리스트
            info_texts_list = []
            description_parts = []

            for i, info_text in enumerate(info_texts):
                clean_text = clean_html(info_text)
                info_texts_list.append(clean_text)
                description_parts.append(clean_text)

            if description_parts:
                description = '\n\n'.join(description_parts)
                print(f"직접 추출로 설명 생성 성공: {len(description)} 바이트")

            info_texts_json = json.dumps(info_texts_list, ensure_ascii=False)
            print(f"직접 추출로 info_texts JSON 생성 성공: {len(info_texts_json)} 바이트")
        else:
            # 컨텍스트 출력 (디버깅용)
            context = find_context(page_source, "intro_bottom")
            if context:
                print(f"intro_bottom 관련 컨텍스트:\n{context}")

            context = find_context(page_source, "info_text")
            if context:
                print(f"info_text 관련 컨텍스트:\n{context}")

    # 결과 반환
    return description, info_texts_json