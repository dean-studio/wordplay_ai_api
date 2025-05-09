import re
from typing import List, Dict, Any


class QuestionParser:
    """문제 파싱 클래스"""

    @staticmethod
    def parse_multiple_choice(text: str) -> List[Dict[str, Any]]:
        """객관식 문제 파싱"""
        mc_questions = []

        # 각 객관식 문제 블록 찾기
        mc_blocks = re.findall(r"## 객관식 문제 \d+(.*?)(?=## 객관식 문제 \d+|## OX|$)", text, re.DOTALL)

        for block in mc_blocks:
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

            question_match = re.search(r"질문:\s*([^\n]+)", block)
            if question_match:
                mc_obj["question"] = question_match.group(1).strip()
            else:
                # 질문이 없으면 이 블록은 스킵
                continue

            options = re.findall(r"보기(\d):\s*([^\n]+)", block)
            for opt in options:
                idx = int(opt[0])
                if 1 <= idx <= 4:
                    mc_obj["options"][idx - 1]["value"] = opt[1].strip()

            answer_match = re.search(r"정답:\s*(\d)", block)
            if answer_match:
                mc_obj["answer"] = int(answer_match.group(1))

            explanation_match = re.search(r"설명:\s*([^\n]+)", block)
            if explanation_match:
                mc_obj["explanation"] = explanation_match.group(1).strip()

            mc_questions.append(mc_obj)

        return mc_questions

    # question_parser.py
    # OX 문제 파싱 함수 수정
    @staticmethod
    def parse_ox_questions(text: str) -> List[Dict[str, Any]]:
        """OX 문제 파싱"""
        ox_questions = []

        # 각 OX 문제 블록 찾기
        ox_blocks = re.findall(r"## OX 문제 \d+|OX 문제 \d+(.*?)(?=## OX 문제 \d+|OX 문제 \d+|$)", text, re.DOTALL)

        for block in ox_blocks:
            if not block.strip():  # 빈 블록 건너뛰기
                continue

            ox_obj = {
                "question": "",
                "answer": True,
                "explanation": ""
            }

            question_match = re.search(r"질문:\s*([^\n]+)", block)
            if question_match:
                ox_obj["question"] = question_match.group(1).strip()
            else:
                # 질문이 없으면 이 블록은 스킵
                continue

            # 정답 부분에서 O, X, Y, N 모두 인식하도록 수정
            answer_match = re.search(r"정답:\s*([OXYNoxyn])", block)
            if answer_match:
                answer_text = answer_match.group(1).upper()
                # O 또는 Y는 True, X 또는 N은 False로 처리
                ox_obj["answer"] = True if answer_text in ['O', 'Y'] else False

            explanation_match = re.search(r"설명:\s*([^\n]+)", block)
            if explanation_match:
                ox_obj["explanation"] = explanation_match.group(1).strip()

            ox_questions.append(ox_obj)

        return ox_questions