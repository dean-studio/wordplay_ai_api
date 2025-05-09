# question_parser.py 수정

import re
from typing import List, Dict, Any


class QuestionParser:
    """문제 파싱 클래스"""

    @staticmethod
    def parse_multiple_choice(text: str) -> List[Dict[str, Any]]:
        """객관식 문제 파싱"""
        mc_questions = []

        # 각 객관식 문제 블록 찾기 (형식 다양성 처리)
        mc_blocks = re.findall(r"(?:## 객관식 문제 \d+|객관식 문제 \d+)(.*?)(?=## 객관식 문제 \d+|객관식 문제 \d+|## OX|OX 문제|$)", text,
                               re.DOTALL)

        for block in mc_blocks:
            if not block.strip():  # 빈 블록 건너뛰기
                continue

            mc_obj = {
                "question": "",
                "options": [
                    {"id": 1, "value": "보기1"},
                    {"id": 2, "value": "보기2"},
                    {"id": 3, "value": "보기3"},
                    {"id": 4, "value": "보기4"}
                ],
                "answer": 1,
                "explanation": ""
            }

            # 질문 찾기
            question_match = re.search(r"질문:\s*([^\n]+)", block)
            if not question_match:
                # 질문이 없으면 다른 패턴 시도
                question_match = re.search(r"\n(.+?)\n[1-4]\.|\n(.+?)\n보기[1-4]:", block)

            if question_match:
                # group(1) 또는 group(2) 중 None이 아닌 것 선택
                if question_match.group(1):
                    mc_obj["question"] = question_match.group(1).strip()
                elif len(question_match.groups()) > 1 and question_match.group(2):
                    mc_obj["question"] = question_match.group(2).strip()
            else:
                # 질문이 없으면 이 블록은 스킵
                continue

            # 선택지 찾기 (두 가지 패턴 모두 확인)
            options_pattern1 = re.findall(r"보기([1-4]):\s*([^\n]+)", block)
            options_pattern2 = re.findall(r"([1-4])[\.|\)]\s*([^\n]+)", block)

            options = options_pattern1 if options_pattern1 else options_pattern2

            for opt in options:
                idx = int(opt[0])
                if 1 <= idx <= 4:
                    mc_obj["options"][idx - 1]["value"] = opt[1].strip()

            # 정답 찾기 (여러 패턴 시도)
            answer_patterns = [
                r"정답:\s*([1-4])[^\d]",  # 정답: 1번
                r"정답:\s*([1-4])번",  # 정답: 1번
                r"정답:\s*([1-4])\s*-",  # 정답: 1 -
                r"정답:\s*([1-4])",  # 정답: 1
                r"정답:\s*보기([1-4])",  # 정답: 보기1
                r"정답은\s*([1-4])번",  # 정답은 1번
                r"정답은\s*([1-4])\s*-",  # 정답은 1 -
                r"정답:\s*([1-4])[^0-9]"  # 정답: 1 (숫자 아닌 문자 뒤에)
            ]

            for pattern in answer_patterns:
                answer_match = re.search(pattern, block)
                if answer_match:
                    mc_obj["answer"] = int(answer_match.group(1))
                    break

            # 설명 찾기 (여러 패턴 시도)
            explanation_patterns = [
                r"설명:\s*([^\n]+)",  # 설명: ...
                r"정답.+?-\s*(.+)",  # 정답 N번 - ...
                r"정답.+?\s+-\s+(.+)",  # 정답 N번 - ...
                r"정답.+?:(.+)",  # 정답 N번: ...
            ]

            for pattern in explanation_patterns:
                explanation_match = re.search(pattern, block)
                if explanation_match:
                    mc_obj["explanation"] = explanation_match.group(1).strip()
                    break

            mc_questions.append(mc_obj)

        return mc_questions

    @staticmethod
    def parse_ox_questions(text: str) -> List[Dict[str, Any]]:
        """OX 문제 파싱"""
        ox_questions = []

        # 각 OX 문제 블록 찾기 (형식 다양성 처리)
        ox_blocks = re.findall(r"(?:## OX 문제 \d+|OX 문제 \d+)(.*?)(?=## OX 문제 \d+|OX 문제 \d+|## 객관식|객관식 문제|$)", text,
                               re.DOTALL)

        for block in ox_blocks:
            if not block.strip():  # 빈 블록 건너뛰기
                continue

            ox_obj = {
                "question": "",
                "answer": True,
                "explanation": ""
            }

            # 질문 찾기
            question_match = re.search(r"질문:\s*([^\n]+)", block)
            if not question_match:
                # 다른 패턴 시도 (첫 줄이 질문일 수 있음)
                lines = block.strip().split('\n')
                if lines and lines[0].strip() and not lines[0].startswith("정답"):
                    ox_obj["question"] = lines[0].strip()
            else:
                ox_obj["question"] = question_match.group(1).strip()

            if not ox_obj["question"]:
                # 질문이 없으면 이 블록은 스킵
                continue

            # 정답 찾기 (다양한 형식 지원)
            answer_patterns = [
                r"정답:\s*([OXYNoxyn])[^\w]",  # 정답: O 또는 X (그 뒤에 비단어문자)
                r"정답:\s*([OXYNoxyn])$",  # 정답: O 또는 X (줄 끝)
                r"정답:\s*([OXYNoxyn])\s",  # 정답: O 또는 X (그 뒤에 공백)
                r"정답은\s*([OXYNoxyn])[^\w]",  # 정답은 O 또는 X
                r"정답은\s*([OXYNoxyn])입니다",  # 정답은 O입니다
                r"정답:\s*([OXYNoxyn])",  # 정답: O 또는 X (일반)
                r"정답\s*[:：]\s*([OXYNoxyn])",  # 정답 : O (콜론 뒤)
            ]

            for pattern in answer_patterns:
                answer_match = re.search(pattern, block)
                if answer_match:
                    answer_text = answer_match.group(1).upper()
                    # O 또는 Y는 True, X 또는 N은 False로 처리
                    ox_obj["answer"] = True if answer_text in ['O', 'Y'] else False
                    break

            # 설명 찾기 (다양한 패턴 지원)
            explanation_patterns = [
                r"설명:\s*([^\n]+)",  # 설명: ...
                r"정답.+?-\s*(.+)",  # 정답 X - ...
                r"정답.+?\s+-\s+(.+)",  # 정답 X - ...
                r"정답.+?:(.+)",  # 정답 X: ...
            ]

            for pattern in explanation_patterns:
                explanation_match = re.search(pattern, block)
                if explanation_match:
                    ox_obj["explanation"] = explanation_match.group(1).strip()
                    break

            ox_questions.append(ox_obj)

        return ox_questions