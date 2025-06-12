#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import requests
from typing import List, Optional
from utils.html_parser import find_context


def extract_preview_pid(page_source: str) -> Optional[str]:
    """미리보기 PID 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        str: 추출된 PID 또는 None
    """
    pid_pattern = r'data-kbbfn-pid="([^"]+)"'
    pid_match = re.search(pid_pattern, page_source)

    if pid_match:
        pid = pid_match.group(1)
        print(f"미리보기 PID 추출 성공: {pid}")
        return pid

    print("미리보기 PID 추출 실패")
    return None


def extract_preview(page_source: str) -> List[str]:
    """미리보기 이미지 전체 추출

    Args:
        page_source (str): 페이지 소스

    Returns:
        List[str]: 미리보기 이미지 URL 목록
    """
    pid = extract_preview_pid(page_source)
    if not pid:
        return []

    preview_page_source = fetch_preview_page(pid)
    if not preview_page_source:
        return []

    return extract_preview_images(preview_page_source)


def fetch_preview_page(pid: str) -> Optional[str]:
    """미리보기 페이지 소스 가져오기

    Args:
        pid (str): 상품 PID

    Returns:
        str: 미리보기 페이지 소스 또는 None
    """
    preview_url = f"https://product.kyobobook.co.kr/book/preview/{pid}"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(preview_url, headers=headers, timeout=10)
        response.raise_for_status()

        print(f"미리보기 페이지 접근 성공: {preview_url}")
        return response.text

    except requests.RequestException as e:
        print(f"미리보기 페이지 접근 실패: {e}")
        return None


def extract_preview_images(preview_page_source: str) -> List[str]:
    """미리보기 이미지 URL 목록 추출

    Args:
        preview_page_source (str): 미리보기 페이지 소스

    Returns:
        List[str]: 이미지 URL 목록
    """
    image_urls = []

    img_pattern = r'<img[^>]*src="([^"]+)"[^>]*alt="[^"]*페이지[^"]*"[^>]*>'
    img_matches = re.findall(img_pattern, preview_page_source)

    for img_url in img_matches:
        if 'prvw' in img_url and img_url not in image_urls:
            image_urls.append(img_url)

    if not image_urls:
        print("기본 패턴 매치 실패, 대체 패턴 시도")

        alt_img_pattern = r'<img[^>]*src="(https://contents\.kyobobook\.co\.kr/prvw/[^"]+)"[^>]*>'
        alt_img_matches = re.findall(alt_img_pattern, preview_page_source)

        for img_url in alt_img_matches:
            if img_url not in image_urls:
                image_urls.append(img_url)

    if not image_urls:
        print("대체 패턴도 실패, 모든 이미지 URL 추출 시도")

        all_img_pattern = r'<img[^>]*src="([^"]+)"[^>]*>'
        all_img_matches = re.findall(all_img_pattern, preview_page_source)

        for img_url in all_img_matches:
            if 'kyobobook.co.kr/prvw' in img_url and img_url not in image_urls:
                image_urls.append(img_url)

    print(f"미리보기 이미지 URL 추출 완료: {len(image_urls)}개")
    for i, url in enumerate(image_urls, 1):
        print(f"  {i}. {url}")

    return image_urls


def extract_all_preview_images(page_source: str) -> List[str]:
    """전체 미리보기 이미지 추출 프로세스

    Args:
        page_source (str): 메인 페이지 소스

    Returns:
        List[str]: 미리보기 이미지 URL 목록
    """
    pid = extract_preview_pid(page_source)
    if not pid:
        return []

    preview_page_source = fetch_preview_page(pid)
    if not preview_page_source:
        return []

    return extract_preview_images(preview_page_source)


def extract_preview_cover_image(page_source: str) -> Optional[str]:
    """미리보기 커버 이미지만 추출

    Args:
        page_source (str): 메인 페이지 소스

    Returns:
        str: 커버 이미지 URL 또는 None
    """
    image_urls = extract_all_preview_images(page_source)

    if image_urls:
        cover_image = image_urls[0]
        print(f"미리보기 커버 이미지: {cover_image}")
        return cover_image

    return None
