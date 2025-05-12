#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def setup_browser():
    """셀레니움 브라우저 설정"""
    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # 새로운 헤드리스 모드 사용
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1280,720")

    # 사용자 에이전트 설정
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # 불필요한 로깅 제거
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    # 페이지 리소스 제한 - 일부 제한 완화
    chrome_options.add_experimental_option('prefs', {
        'profile.default_content_setting_values': {
            'plugins': 2,  # 플러그인 차단
            'popups': 2,  # 팝업 차단
            'geolocation': 2,  # 위치 차단
            'notifications': 2,  # 알림 차단
            'auto_select_certificate': 2,  # 인증서 선택 차단
            'fullscreen': 2,  # 전체 화면 차단
            'mouselock': 2,  # 마우스 잠금 차단
            'mixed_script': 2,  # 혼합 스크립트 차단
            'media_stream': 2,  # 미디어 스트림 차단
            'media_stream_mic': 2,  # 마이크 차단
            'media_stream_camera': 2,  # 카메라 차단
            'protocol_handlers': 2,  # 프로토콜 핸들러 차단
            'ppapi_broker': 2,  # PPAPI 브로커 차단
            'automatic_downloads': 2,  # 자동 다운로드 차단
            'midi_sysex': 2,  # MIDI 차단
            'push_messaging': 2,  # 푸시 메시지 차단
            'ssl_cert_decisions': 2,  # SSL 인증서 결정 차단
            'metro_switch_to_desktop': 2,  # 메트로 전환 차단
            'protected_media_identifier': 2,  # 보호된 미디어 식별자 차단
            'app_banner': 2,  # 앱 배너 차단
            'site_engagement': 2,  # 사이트 참여 차단
            'durable_storage': 2  # 내구성 있는 스토리지 차단
        }
    })

    # 드라이버 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # 타임아웃 설정
    driver.set_page_load_timeout(360)  # 페이지 로드 타임아웃 360초

    return driver