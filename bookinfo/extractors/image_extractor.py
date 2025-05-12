#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from utils.html_parser import find_context


def extract_cover_image(page_source):
    """커버 이미지 URL 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        str: 추출된 이미지 URL 또는 None
    """
    # 커버 이미지 패턴 매칭
    image_pattern = r'<div[^>]*class="portrait_img_box[^"]*"[^>]*>.*?<img[^>]*src="([^"]+)"[^>]*>'
    image_match = re.search(image_pattern, page_source, re.DOTALL)

    if image_match:
        image_url = image_match.group(1)
        print(f"커버 이미지 URL 추출 성공: {image_url}")
        return image_url

    print("커버 이미지 URL 패턴 매치 실패, 대체 패턴 시도")

    # 대체 패턴
    alt_image_pattern = r'<img[^>]*alt="[^"]*대표[^"]*"[^>]*src="([^"]+)"[^>]*>'
    alt_image_match = re.search(alt_image_pattern, page_source, re.DOTALL)

    if alt_image_match:
        image_url = alt_image_match.group(1)
        print(f"커버 이미지 URL 추출 성공 (대체 패턴): {image_url}")
        return image_url

    # 컨텍스트 출력 (디버깅용)
    image_context = find_context(page_source, 'portrait_img_box')
    if image_context:
        print(f"이미지 관련 HTML 부분:\n{image_context}")

    print("커버 이미지 URL 추출 실패")
    return None


def extract_description_image(page_source):
    """추가 설명 이미지 URL 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        str: 추출된 이미지 URL 또는 None
    """
    # 추가 설명 이미지 패턴 매칭
    desc_image_pattern = r'<div[^>]*class="product_detail_area detail_img"[^>]*>.*?<img[^>]*src="([^"]+)"[^>]*>'
    desc_image_match = re.search(desc_image_pattern, page_source, re.DOTALL)

    if desc_image_match:
        image_url = desc_image_match.group(1)
        print(f"추가 설명 이미지 URL 추출 성공: {image_url}")
        return image_url

    print("추가 설명 이미지 URL 패턴 매치 실패, 대체 패턴 시도")

    # 대체 패턴
    alt_desc_image_pattern = r'<div[^>]*class="product_detail_area[^"]*"[^>]*>.*?<div[^>]*class="inner"[^>]*>.*?<img[^>]*src="([^"]+)"[^>]*>'
    alt_desc_image_match = re.search(alt_desc_image_pattern, page_source, re.DOTALL)

    if alt_desc_image_match:
        image_url = alt_desc_image_match.group(1)
        print(f"추가 설명 이미지 URL 추출 성공 (대체 패턴): {image_url}")
        return image_url

    # 컨텍스트 출력 (디버깅용)
    detail_img_context = find_context(page_source, 'detail_img')
    if detail_img_context:
        print(f"상세 이미지 관련 HTML 부분:\n{detail_img_context}")

    print("추가 설명 이미지 URL 추출 실패")
    return None