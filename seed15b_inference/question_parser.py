import re
from typing import List, Dict, Any


class QuestionParser:
    """문제 파싱 클래스"""

    @staticmethod
    def parse_multiple_choice(text: str) -> List[Dict[str, Any]]:
        """객관식 문제 파싱 - 패턴 확장 버전"""
        print("=== 객관식 문제 파싱 입력 ===")
        print(text)

        mc_questions = []

        # 각 객관식 문제 블록 찾기 (형식 다양성 처리)
        mc_blocks = re.findall(r"(?:## )?객관식 문제 (\d+)(.*?)(?=(?:## )?객관식 문제 \d+|(?:## )?OX|$)", text,
                               re.DOTALL | re.IGNORECASE)

        print(f"찾은 객관식 블록 수: {len(mc_blocks)}")

        # 블록이 없으면 질문과 선택지 패턴으로 직접 분리 시도
        if not mc_blocks:
            # 질문과 선택지 패턴 찾기
            mc_parts = re.split(r'질문:|문제:', text)
            if len(mc_parts) > 1:
                for i in range(1, len(mc_parts)):
                    mc_blocks.append((str(i), mc_parts[i]))

        for block_tuple in mc_blocks:
            block_num, block = block_tuple

            if not block.strip():  # 빈 블록 건너뛰기
                continue

            print(f"=== 처리 중인 객관식 블록 {block_num} ===\n{block}\n==========")

            mc_obj = {
                "question": "",
                "options": [
                    {"id": 1, "value": ""},
                    {"id": 2, "value": ""},
                    {"id": 3, "value": ""},
                    {"id": 4, "value": ""}
                ],
                "answer": 1,
                "explanation": ""
            }

            # 질문 찾기
            question_lines = []
            lines = block.strip().split('\n')
            for i, line in enumerate(lines):
                if any(marker in line for marker in ['A)', 'A.', 'A:', '보기1:', '1)']) or \
                        re.search(r'^[A-D][\.:\)]', line.strip()):
                    break
                if i > 0 or '질문:' not in block:  # 첫 줄은 이미 질문 마커가 제거된 상태일 수 있음
                    question_lines.append(line.strip())

            if question_lines:
                mc_obj["question"] = ' '.join(question_lines).strip()
            else:
                # 질문을 찾지 못했으면 직접 패턴 찾기
                question_match = re.search(r"질문:\s*([^\n]+)", block)
                if question_match:
                    mc_obj["question"] = question_match.group(1).strip()
                else:
                    # 다른 패턴 시도 (첫 줄이 질문일 수 있음)
                    if lines and lines[0].strip():
                        mc_obj["question"] = lines[0].strip()

            # 질문에서 "질문:" 접두사 제거
            mc_obj["question"] = re.sub(r'^질문:\s*', '', mc_obj["question"])

            # 선택지 찾기 - 다양한 형식 지원
            options = []

            # A), B), C), D) 또는 A., B., C., D. 형식
            option_patterns = [
                r'([A-D])[\)\.:]?\s*(.*?)(?=(?:[A-D][\)\.:])|정답:|$)',  # A) 내용 또는 A. 내용
                r'(\d+)[\)\.:]?\s*(.*?)(?=(?:\d+[\)\.:])|정답:|$)',  # 1) 내용 또는 1. 내용
                r'보기(\d+):\s*(.*?)(?=보기\d+:|정답:|$)'  # 보기1: 내용
            ]

            for pattern in option_patterns:
                opts = re.findall(pattern, block, re.DOTALL)
                if opts:
                    # A-D를 1-4로 변환
                    for i, opt in enumerate(opts):
                        key, value = opt
                        if key in ['A', 'B', 'C', 'D']:
                            idx = ord(key) - ord('A') + 1
                        else:
                            idx = int(key)

                        if 1 <= idx <= 4:
                            options.append((idx, value.strip()))

                    # 선택지를 찾았으면 반복 중단
                    if options:
                        break

            # 선택지 설정
            for idx, value in options:
                if 1 <= idx <= 4:
                    mc_obj["options"][idx - 1]["value"] = value

            # 정답 찾기
            answer_pattern = r'정답\s*:?\s*([A-D1-4])'
            answer_match = re.search(answer_pattern, block, re.IGNORECASE)

            if answer_match:
                answer_key = answer_match.group(1)
                if answer_key in ['A', 'B', 'C', 'D']:
                    mc_obj["answer"] = ord(answer_key) - ord('A') + 1
                else:
                    mc_obj["answer"] = int(answer_key)

            # 설명 찾기
            explanation_pattern = r'설명\s*:?\s*(.*?)(?=$)'
            explanation_match = re.search(explanation_pattern, block, re.DOTALL | re.IGNORECASE)

            if explanation_match:
                mc_obj["explanation"] = explanation_match.group(1).strip()
            else:
                # 정답 다음 줄부터의 내용을 설명으로 간주
                answer_line_idx = -1
                for i, line in enumerate(lines):
                    if '정답' in line:
                        answer_line_idx = i
                        break

                if answer_line_idx >= 0 and answer_line_idx + 1 < len(lines):
                    explanation_lines = [lines[i].strip() for i in range(answer_line_idx + 1, len(lines)) if
                                         lines[i].strip()]
                    if explanation_lines:
                        mc_obj["explanation"] = ' '.join(explanation_lines)

            # 기본값 처리
            if not mc_obj["question"]:
                continue  # 질문이 없으면 이 문제는 건너뜀

            # 선택지 기본값 처리
            empty_options = True
            for opt in mc_obj["options"]:
                if opt["value"]:
                    empty_options = False
                    break

            if empty_options:
                # 선택지가 비어있다면 A, B 등을 인식하지 못한 것일 수 있음
                alphabet_options = re.findall(r'([A-D])[\.:\)]?\s*([^\n]+)', block)
                if alphabet_options:
                    for letter, content in alphabet_options:
                        idx = ord(letter) - ord('A') + 1
                        if 1 <= idx <= 4:
                            mc_obj["options"][idx - 1]["value"] = content.strip()
                else:
                    # 숫자로 된 선택지 찾기
                    number_options = re.findall(r'(\d+)[\.:\)]?\s*([^\n]+)', block)
                    for num_str, content in number_options:
                        try:
                            idx = int(num_str)
                            if 1 <= idx <= 4:
                                mc_obj["options"][idx - 1]["value"] = content.strip()
                        except ValueError:
                            pass

            # 여전히 선택지가 비어있다면 기본값 설정
            empty_options = True
            for opt in mc_obj["options"]:
                if opt["value"]:
                    empty_options = False
                    break

            if empty_options:
                for i in range(4):
                    mc_obj["options"][i]["value"] = f"보기{i + 1}"

            # 설명이 없으면 기본 설명 추가
            if not mc_obj["explanation"]:
                mc_obj["explanation"] = f"정답은 {mc_obj['answer']}번입니다."

            mc_questions.append(mc_obj)

        print(f"최종 파싱된 객관식 문제 수: {len(mc_questions)}")

        # 문제가 하나도 파싱되지 않은 경우 기본 문제 생성
        if not mc_questions:
            print("객관식 문제를 파싱하지 못했습니다. 기본 문제를 생성합니다.")
            mc_questions.append({
                "question": "주어진 내용에 기반한 객관식 문제",
                "options": [
                    {"id": 1, "value": "보기1"},
                    {"id": 2, "value": "보기2"},
                    {"id": 3, "value": "보기3"},
                    {"id": 4, "value": "보기4"}
                ],
                "answer": 1,
                "explanation": "문제 파싱에 실패하여 기본 문제를 생성했습니다."
            })

        return mc_questions

    @staticmethod
    def parse_ox_questions(text: str) -> List[Dict[str, Any]]:
        """OX 문제 파싱 - 패턴 확장 버전"""
        print("=== OX 문제 파싱 입력 ===")
        print(text)

        ox_questions = []

        # 텍스트가 비어있거나 None인 경우 빈 리스트 반환
        if not text:
            print("OX 문제 텍스트가 비어 있습니다.")
            return []

        # 더 다양한 패턴 시도 (케이스 민감도 낮춤)
        patterns = [
            r"(?:## )?[Oo][Xx] 문제 (\d+)(.*?)(?=(?:## )?[Oo][Xx] 문제 \d+|$)",  # OX 문제 1 또는 ## OX 문제 1
            r"(?:## )?[Oo][Xx]퀴즈 (\d+)(.*?)(?=(?:## )?[Oo][Xx]퀴즈 \d+|$)",  # OX퀴즈 1 또는 ## OX퀴즈 1
            r"(?:## )?[Oo]/[Xx] 문제 (\d+)(.*?)(?=(?:## )?[Oo]/[Xx] 문제 \d+|$)"  # O/X 문제 1 또는 ## O/X 문제 1
        ]

        # 여러 패턴 시도
        all_blocks = []
        for pattern in patterns:
            blocks = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if blocks:
                # 결과가 튜플 (번호, 내용)으로 반환됨
                all_blocks.extend([block[1] for block in blocks])

        print(f"찾은 OX 블록 수: {len(all_blocks)}")

        # 블록이 없으면 다른 방식으로 시도 (단순히 문제 별로 분리)
        if not all_blocks and ("진술:" in text or "정답:" in text or "해설:" in text):
            # 진술, 정답, 해설 패턴을 기준으로 분리
            sections = re.split(r'(진술:|정답:|해설:)', text)
            if len(sections) > 1:
                # 재구성
                current_block = ""
                for i, section in enumerate(sections):
                    current_block += section
                    if i > 0 and i % 6 == 5:  # 진술, 정답, 해설이 각각 2개씩 나오면 (텍스트+키워드)
                        all_blocks.append(current_block)
                        current_block = ""
                if current_block:  # 남은 블록 처리
                    all_blocks.append(current_block)

        # 여전히 블록이 없으면 텍스트 전체를 하나의 블록으로 처리
        if not all_blocks and ("Y" in text or "N" in text or "O" in text or "X" in text):
            print("전체 텍스트를 OX 블록으로 처리합니다.")
            all_blocks = [text]

        for block in all_blocks:
            if not block.strip():  # 빈 블록 건너뛰기
                continue

            print(f"=== 처리 중인 OX 블록 ===\n{block}\n==========")

            ox_obj = {
                "question": "",
                "answer": True,
                "explanation": ""
            }

            # 진술문 찾기 (다양한 패턴)
            statement_patterns = [
                r"진술:\s*([^\n]+)",
                r"질문:\s*([^\n]+)",
                r"문제:\s*([^\n]+)"
            ]

            for pattern in statement_patterns:
                statement_match = re.search(pattern, block, re.IGNORECASE)
                if statement_match:
                    ox_obj["question"] = statement_match.group(1).strip()
                    break

            # 진술문이 아직 없으면 줄 별로 확인
            if not ox_obj["question"]:
                lines = block.strip().split('\n')
                for i, line in enumerate(lines):
                    # 정답, 해설 등의 키워드가 없고, Y/N/O/X도 없는 첫 줄일 가능성이 높음
                    if i == 0 and not any(kw in line.lower() for kw in ["정답", "해설", "설명"]):
                        if not re.search(r'^\s*[YNOXynox]\s*$', line):
                            ox_obj["question"] = line.strip()
                            break

            # 여전히 진술문이 없으면 계속 탐색
            if not ox_obj["question"]:
                for i, line in enumerate(block.strip().split('\n')):
                    if line and ":" not in line and not re.search(r'^\s*[YNOXynox]\s*$', line):
                        ox_obj["question"] = line.strip()
                        break

            # 정답 찾기 (Y/N/O/X)
            answer_patterns = [
                r"정답:\s*([YNOXynox])",
                r"답:\s*([YNOXynox])",
                r"^([YNOXynox])$"  # 한 줄에 Y/N/O/X만 있는 경우
            ]

            for pattern in answer_patterns:
                answer_match = re.search(pattern, block, re.IGNORECASE)
                if answer_match:
                    answer_text = answer_match.group(1).upper()
                    ox_obj["answer"] = True if answer_text in ['Y', 'O'] else False
                    break

            # 설명 찾기
            explanation_patterns = [
                r"해설:\s*([^\n]+)",
                r"설명:\s*([^\n]+)",
                r"이유:\s*([^\n]+)"
            ]

            for pattern in explanation_patterns:
                explanation_match = re.search(pattern, block, re.IGNORECASE)
                if explanation_match:
                    ox_obj["explanation"] = explanation_match.group(1).strip()
                    break

            # 설명이 없으면 정답 다음 줄부터의 내용을 설명으로 간주
            if not ox_obj["explanation"]:
                lines = block.strip().split('\n')
                for i, line in enumerate(lines):
                    if "정답:" in line and i + 1 < len(lines):
                        ox_obj["explanation"] = ' '.join([l.strip() for l in lines[i + 1:] if l.strip()])
                        break

            # 질문이 없으면 설명에서 추출 시도
            if not ox_obj["question"] and ox_obj["explanation"]:
                ox_obj[
                    "question"] = f"다음 진술은 {ox_obj['answer'] and '참' or '거짓'}이다: {ox_obj['explanation'].split('.')[0]}."

            # 기본값 설정 (처리된 항목이 없을 경우)
            if not ox_obj["question"]:
                continue  # 질문이 없으면 이 블록은 건너뜀

            # 설명이 없으면 기본 설명 추가
            if not ox_obj["explanation"]:
                ox_obj["explanation"] = f"이 진술은 {'참입니다' if ox_obj['answer'] else '거짓입니다'}."

            ox_questions.append(ox_obj)

        print(f"최종 파싱된 OX 문제 수: {len(ox_questions)}")

        # 문제가 하나도 파싱되지 않은 경우 텍스트 전체에서 핵심 정보 추출 시도
        if not ox_questions:
            print("텍스트 전체에서 OX 문제 정보 추출 시도...")
            # Y/N 또는 O/X가 있는 부분 찾기
            yn_sections = re.findall(r'([^\n]+)[^\w]*[YNOXynox][^\w]*([^\n]+)', text)

            for section in yn_sections:
                statement = section[0].strip()
                explanation = section[1].strip()

                # Y/N 또는 O/X 찾기
                answer_match = re.search(r'[YNOXynox]', statement + explanation)
                if answer_match:
                    answer_text = answer_match.group(0).upper()
                    answer = True if answer_text in ['Y', 'O'] else False

                    # 진술문 정리 (Y/N/O/X 및 관련 텍스트 제거)
                    statement = re.sub(r'[YNOXynox]', '', statement).strip()
                    statement = re.sub(r'정답:|\s+정답\s+|해설:|진술:', '', statement).strip()

                    if statement:
                        ox_questions.append({
                            "question": statement,
                            "answer": answer,
                            "explanation": explanation
                        })

            print(f"전체 텍스트 분석으로 찾은 OX 문제 수: {len(ox_questions)}")

        # 여전히 문제가 없으면 기본 문제 생성
        if not ox_questions:
            print("OX 문제를 파싱하지 못했습니다. 강제 추출을 시도합니다.")

            # 강제 추출: 텍스트에서 Y/N 또는 O/X가 있는 문장 찾기
            answer_markers = ['Y', 'N', 'O', 'X']
            for marker in answer_markers:
                if marker in text.upper():
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        if marker in line.upper() and i > 0:
                            statement = lines[i - 1].strip()
                            explanation = ' '.join([l.strip() for l in lines[i + 1:i + 3] if l.strip()])
                            if statement and not any(kw in statement.lower() for kw in ["문제", "진술", "질문", "정답", "해설"]):
                                ox_questions.append({
                                    "question": statement,
                                    "answer": True if marker in ['Y', 'O'] else False,
                                    "explanation": explanation if explanation else f"정답은 {marker}입니다."
                                })
                                break

            print(f"강제 추출로 찾은 OX 문제 수: {len(ox_questions)}")

        # 여전히 문제가 없으면 기본 문제 사용
        if not ox_questions:
            print("모든 방법으로 OX 문제 추출 실패. 기본 문제를 생성합니다.")
            ox_questions.append({
                "question": "이 교육 콘텐츠는 중요한 정보를 포함하고 있다.",
                "answer": True,
                "explanation": "모든 교육 콘텐츠는 학습에 중요한 정보를 담고 있습니다."
            })

        return ox_questions