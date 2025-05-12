#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging


def setup_logger():
    """로거 설정"""
    logger = logging.getLogger('kyobobook_scraper')
    logger.setLevel(logging.INFO)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포맷터
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(console_handler)

    return logger


# 로거 인스턴스 생성
logger = setup_logger()