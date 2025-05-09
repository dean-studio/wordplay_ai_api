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
        print("=== OX 문제 파싱 입력 ===")
        print(text)

        ox_questions = []

        # 텍스트가 비어있거나 None인 경우 빈 리스트 반환
        if not text:
            print("OX 문제 텍스트가 비어 있습니다.")
            return []

        # 각 OX 문제 블록 찾기 (형식 다양성 처리)
        ox_blocks = re.findall(r"(?:## OX 문제 \d+|OX 문제 \d+)(.*?)(?=## OX 문제 \d+|OX 문제 \d+|## 객관식|객관식 문제|$)", text,
                               re.DOTALL)

        print(f"찾은 OX 블록 수: {len(ox_blocks)}")

        # 블록이 없으면 다른 패턴 시도
        if not ox_blocks:
            # 다른 정규식 패턴 시도
            ox_blocks = re.findall(r"(?:OX 문제|O/X 문제|OX 퀴즈).*?(?=(?:OX 문제|O/X 문제|OX 퀴즈)|$)", text, re.DOTALL)
            print(f"대체 패턴으로 찾은 OX 블록 수: {len(ox_blocks)}")

            # 여전히 블록이 없으면 텍스트 전체를 하나의 블록으로 처리
            if not ox_blocks and ("O" in text or "X" in text):
                ox_blocks = [text]
                print("전체 텍스트를 하나의 OX 블록으로 처리합니다.")

        for block in ox_blocks:
            if not block.strip():  # 빈 블록 건너뛰기
                continue

            print(f"=== 처리 중인 OX 블록 ===\n{block}\n==========")

            ox_obj = {
                "question": "",
                "answer": True,
                "explanation": ""
            }

            # 질문 찾기
            question_match = re.search(r"질문:\s*([^\n]+)", block)
            if not question_match:
                # 다른 패턴 시도
                question_match = re.search(r"question:\s*([^\n]+)", block, re.IGNORECASE)

            if question_match:
                # "question: " 또는 "질문: " 접두사 제거
                question_text = question_match.group(1).strip()
                question_text = re.sub(r'^(question:|질문:)\s*', '', question_text, flags=re.IGNORECASE)
                ox_obj["question"] = question_text
            else:
                # 다른 패턴 시도 (첫 줄이 질문일 수 있음)
                lines = block.strip().split('\n')
                if lines and lines[0].strip() and not lines[0].startswith("정답"):
                    # "question: " 또는 "질문: " 접두사 제거
                    first_line = lines[0].strip()
                    first_line = re.sub(r'^(question:|질문:)\s*', '', first_line, flags=re.IGNORECASE)
                    ox_obj["question"] = first_line

            if not ox_obj["question"]:
                # 질문이 없으면 블럭 전체에서 질문 형태 찾기
                question_candidates = re.findall(r"([^.\n]+\?|[^.\n]+은 [OXox]입니다)", block)
                if question_candidates:
                    first_candidate = question_candidates[0].strip()
                    first_candidate = re.sub(r'^(question:|질문:)\s*', '', first_candidate, flags=re.IGNORECASE)
                    ox_obj["question"] = first_candidate
                else:
                    # 여전히 질문이 없으면 이 블록은 스킵
                    print("질문을 찾을 수 없어 해당 블록을 건너뜁니다.")
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
                r"^\s*([OXYNoxyn])\s*$",  # 줄에 O 또는 X만 있는 경우
                r"answer:\s*([OXYNoxyn])",  # answer: O 또는 X
            ]

            answer_found = False
            for pattern in answer_patterns:
                answer_match = re.search(pattern, block, re.IGNORECASE)
                if answer_match:
                    answer_text = answer_match.group(1).upper()
                    # O 또는 Y는 True, X 또는 N은 False로 처리
                    ox_obj["answer"] = True if answer_text in ['O', 'Y'] else False
                    answer_found = True
                    break

            # 정답을 찾지 못했으면 "O" 또는 "X"가 있는지 확인
            if not answer_found:
                if "X" in block.upper() or "FALSE" in block.upper() or "거짓" in block:
                    ox_obj["answer"] = False
                else:
                    # 기본값은 True
                    ox_obj["answer"] = True

            # 설명 찾기 (다양한 패턴 지원)
            explanation_patterns = [
                r"설명:\s*([^\n]+)",  # 설명: ...
                r"정답.+?-\s*(.+)",  # 정답 X - ...
                r"정답.+?\s+-\s+(.+)",  # 정답 X - ...
                r"정답.+?:(.+)",  # 정답 X: ...
                r"이유:\s*([^\n]+)",  # 이유: ...
                r"해설:\s*([^\n]+)",  # 해설: ...
                r"explanation:\s*([^\n]+)",  # explanation: ...
            ]

            for pattern in explanation_patterns:
                explanation_match = re.search(pattern, block, re.IGNORECASE)
                if explanation_match:
                    explanation_text = explanation_match.group(1).strip()
                    # "explanation: " 접두사 제거
                    explanation_text = re.sub(r'^(explanation:|설명:)\s*', '', explanation_text, flags=re.IGNORECASE)
                    ox_obj["explanation"] = explanation_text
                    break

            # 설명을 찾지 못했으면 질문과 정답 외의 텍스트를 설명으로
            if not ox_obj["explanation"]:
                # 질문과 정답 라인을 제외한 나머지 텍스트
                lines = block.strip().split('\n')
                explanation_lines = []
                for line in lines:
                    if (not re.search(r'^(질문:|question:)', line, re.IGNORECASE) and
                            not re.search(r'^(정답:|answer:)', line, re.IGNORECASE) and
                            not re.match(r'^\s*[OXYNoxyn]\s*$', line)):
                        # "explanation: " 접두사 제거
                        line = re.sub(r'^(explanation:|설명:)\s*', '', line, flags=re.IGNORECASE)
                        explanation_lines.append(line.strip())

                if explanation_lines:
                    ox_obj["explanation"] = " ".join(explanation_lines)

            # 여전히 설명이 없으면 기본 설명 추가
            if not ox_obj["explanation"]:
                ox_obj["explanation"] = f"해당 문제의 정답은 {'O' if ox_obj['answer'] else 'X'}입니다."

            # 설명에서 참고 자료/링크 부분 제거
            if "```" in ox_obj["explanation"]:
                ox_obj["explanation"] = ox_obj["explanation"].split("```")[0].strip()

            # 설명에 있는 answer/explanation 태그 제거
            ox_obj["explanation"] = re.sub(r'answer:\s*[OXYNoxyn]', '', ox_obj["explanation"], flags=re.IGNORECASE)
            ox_obj["explanation"] = re.sub(r'explanation:\s*', '', ox_obj["explanation"], flags=re.IGNORECASE)

            ox_questions.append(ox_obj)

        print(f"최종 파싱된 OX 문제 수: {len(ox_questions)}")

        # 문제가 하나도 파싱되지 않은 경우 기본 문제 생성
        if not ox_questions:
            print("OX 문제를 파싱하지 못했습니다. 기본 문제를 생성합니다.")
            ox_questions.append({
                "question": "주어진 내용에 기반한 OX 문제",
                "answer": True,
                "explanation": "문제 파싱에 실패하여 기본 문제를 생성했습니다."
            })

        return ox_questions