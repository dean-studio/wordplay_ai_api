import json
import re  # 이 줄 추가
from typing import List, Dict, Any, Optional
from datetime import datetime
from model_manager import ModelManager
from content_analyzer import ContentAnalyzer
from question_generator import QuestionGenerator
from question_parser import QuestionParser


class QuizGenerator:
    """퀴즈 생성 통합 클래스"""

    def __init__(self):
        # 모델 매니저 초기화
        self.model_manager = ModelManager(
            model_id="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
            lora_paths=[
                "../seed15b_lora/clova-lora-qa-final"
            ]
        )

        # 하위 클래스 초기화
        self.analyzer = ContentAnalyzer(self.model_manager)
        self.question_generator = QuestionGenerator(self.model_manager)
        self.parser = QuestionParser()

    def post_process_questions(self, questions: List[Dict], current_date=None):
        """시간 표현 등을 구체적으로 변환하는 후처리 함수"""
        if current_date is None:
            current_date = datetime.now()

        # 상대적 시간 표현 사전
        time_mappings = {
            "작년": f"{current_date.year - 1}년",
            "올해": f"{current_date.year}년",
            "내년": f"{current_date.year + 1}년",
            "지난달": f"{current_date.year}년 {current_date.month - 1 if current_date.month > 1 else 12}월",
            "이번달": f"{current_date.year}년 {current_date.month}월",
            "최근": f"{current_date.year}년 기준",
            "요즘": f"{current_date.year}년 {current_date.month}월 현재",
        }

        # 후처리된 결과 저장할 리스트
        processed_questions = []

        for question in questions:
            # JSON 문자열로 변환해서 일괄 치환
            question_str = json.dumps(question, ensure_ascii=False)

            # 상대적 시간 표현을 구체적인 날짜로 대체
            for rel_time, abs_time in time_mappings.items():
                question_str = question_str.replace(rel_time, abs_time)

            # 다시 JSON 객체로 변환해서 리스트에 추가
            processed_questions.append(json.loads(question_str))

        return processed_questions

    def parse_keywords(self, keywords_text: str) -> List[str]:
        """핵심개념 텍스트를 리스트로 변환"""
        # 목록 형식으로 되어 있는 경우 (예: "- 항목1, 항목2\n- 항목3")
        if "-" in keywords_text:
            items = []
            for line in keywords_text.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    line = line[2:].strip()  # "- " 제거
                    # 쉼표로 구분된 아이템이 있으면 분리
                    if "," in line:
                        items.extend([item.strip() for item in line.split(",")])
                    else:
                        items.append(line)
            return items
        # 쉼표로만 구분된 경우 (예: "항목1, 항목2, 항목3")
        elif "," in keywords_text:
            return [item.strip() for item in keywords_text.split(",")]
        # 단일 항목인 경우
        else:
            return [keywords_text.strip()]

    def generate_quiz(self, content: str, mc_count: int = 2, ox_count: int = 2) -> str:
        """퀴즈 생성 파이프라인 실행"""
        try:
            # 현재 날짜 정보 준비
            current_date = datetime.now()
            date_info = f"{current_date.year}년 {current_date.month}월 {current_date.day}일"

            # 1. 내용 분석
            analysis = self.analyzer.analyze(content)
            print("=== 분석 결과 ===")
            print(analysis)

            # 분석 결과 파싱
            analysis_data = {}

            # 카테고리 파싱
            category_match = re.search(r"카테고리:\s*(.+?)(?:\n|$)", analysis)
            if category_match:
                analysis_data["category"] = category_match.group(1).strip()
            else:
                analysis_data["category"] = "기타"

            # 핵심개념 파싱 - 여러 줄과 목록 형식 지원
            keywords_section = re.search(r"핵심개념:(.*?)(?:시기정보:|$)", analysis, re.DOTALL)
            if keywords_section:
                keywords_text = keywords_section.group(1).strip()
                analysis_data["keywords"] = self.parse_keywords(keywords_text)
            else:
                analysis_data["keywords"] = []

            # 시기정보 파싱
            timeinfo_match = re.search(r"시기정보:\s*(.+?)(?:\n|$)", analysis)
            if timeinfo_match:
                analysis_data["time_info"] = timeinfo_match.group(1).strip()
            else:
                analysis_data["time_info"] = "확인 불가"

            # 날짜 정보 추가
            analysis_data["current_date"] = date_info

            # 분석 결과에 현재 날짜 정보 추가
            analysis_with_date = f"{analysis}\n현재 날짜: {date_info}"

            # 2. 객관식 문제 생성 (날짜 정보 포함)
            mc_questions_text = self.question_generator.generate_multiple_choice(
                content, analysis_with_date, mc_count
            )
            print("=== 생성된 객관식 문제 ===")
            print(mc_questions_text)

            # 3. OX 문제 생성 (날짜 정보 포함)
            ox_questions_text = self.question_generator.generate_ox_questions(
                content, analysis_with_date, ox_count
            )
            print("=== 생성된 OX 문제 ===")
            print(ox_questions_text)

            # 5. 파싱된 문제 후처리
            # mc_questions = self.parse_multiple_choice(mc_questions_text)
            # print("=== 파싱된 객관식 문제 (원본) ===")
            # print(json.dumps(mc_questions, ensure_ascii=False, indent=2))
            #
            # ox_questions = self.parse_ox_questions(ox_questions_text)
            # print("=== 파싱된 OX 문제 (원본) ===")
            # print(json.dumps(ox_questions, ensure_ascii=False, indent=2))

            # 문제 후처리
            # mc_questions = self.clean_multiple_choice(mc_questions)
            # print("=== 후처리된 객관식 문제 ===")
            # print(json.dumps(mc_questions, ensure_ascii=False, indent=2))
            #
            # mc_questions = self.post_process_questions(mc_questions, current_date)
            # ox_questions = self.post_process_questions(ox_questions, current_date)

            # 6. 최종 결과 구성
            result = {
                "analysis": analysis_data,  # 분석 결과 포함
                "questions": []
            }

            # 객관식 문제 추가 (최대 허용 개수까지)
            for i, mc in enumerate(mc_questions[:mc_count]):
                result["questions"].append({
                    "multiple_choice": mc
                })

            # 객관식 문제가 없거나 부족하면 기본 문제 추가
            while len(result["questions"]) < mc_count:
                result["questions"].append({
                    "multiple_choice": {
                        "question": "객관식 문제",
                        "options": [
                            {"id": 1, "value": "보기1"},
                            {"id": 2, "value": "보기2"},
                            {"id": 3, "value": "보기3"},
                            {"id": 4, "value": "보기4"}
                        ],
                        "answer": 1,
                        "explanation": f"정답은 '보기1'입니다. 이유: 기본 설명입니다. (현재 날짜: {date_info})"
                    }
                })

            # OX 문제 추가 (최대 허용 개수까지)
            for i, ox in enumerate(ox_questions[:ox_count]):
                result["questions"].append({
                    "ox_quiz": ox
                })

            # OX 문제가 없거나 부족하면 기본 문제 추가
            while len(result["questions"]) - mc_count < ox_count:
                result["questions"].append({
                    "ox_quiz": {
                        "question": "OX 퀴즈 질문",
                        "answer": True,
                        "explanation": f"정답은 O입니다. 이유: 기본 설명입니다. (현재 날짜: {date_info})"
                    }
                })

            print("=== 최종 JSON ===")
            print(json.dumps(result, ensure_ascii=False, indent=2))

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"오류 발생: {str(e)}")
            # 오류 발생 시 기본 JSON 반환
            return self._generate_fallback_json(mc_count, ox_count)

    def _generate_fallback_json(self, mc_count: int, ox_count: int) -> str:
        """오류 발생 시 기본 JSON 반환"""
        current_date = datetime.now()
        date_info = f"{current_date.year}년 {current_date.month}월 {current_date.day}일"

        result = {
            "analysis": {
                "category": "기타",
                "keywords": ["오류", "기본값", "자동생성"],
                "time_info": "확인 불가",
                "current_date": date_info
            },
            "questions": []
        }

        # 기본 객관식 문제 추가
        for i in range(mc_count):
            result["questions"].append({
                "multiple_choice": {
                    "question": f"객관식 문제 {i + 1}",
                    "options": [
                        {"id": 1, "value": "보기1"},
                        {"id": 2, "value": "보기2"},
                        {"id": 3, "value": "보기3"},
                        {"id": 4, "value": "보기4"}
                    ],
                    "answer": 1,
                    "explanation": f"정답은 '보기1'입니다. 이유: 시스템 오류로 기본 응답이 생성되었습니다. (현재 날짜: {date_info})"
                }
            })

        # 기본 OX 문제 추가
        for i in range(ox_count):
            result["questions"].append({
                "ox_quiz": {
                    "question": f"OX 퀴즈 질문 {i + 1}",
                    "answer": i % 2 == 0,  # 홀수 번째는 True, 짝수 번째는 False
                    "explanation": f"정답은 {'O' if i % 2 == 0 else 'X'}입니다. 이유: 시스템 오류로 기본 응답이 생성되었습니다. (현재 날짜: {date_info})"
                }
            })

        return json.dumps(result, ensure_ascii=False, indent=2)

    def fix_ox_questions(self, ox_questions: List[Dict]) -> List[Dict]:
        """OX 문제가 객관식 형태로 생성된 경우 참/거짓 형태로 수정"""
        fixed_questions = []

        for question in ox_questions:
            fixed_question = question.copy()

            # 객관식 형태의 질문인지 확인
            if any(pattern in question["question"].lower() for pattern in
                   ["다음 중", "무엇인가", "고르시오", "선택하시오", "선택하세요", "올바른 것은", "옳은 것은", "맞는 것은"]):
                # 객관식 형태의 질문을 진술문으로 변환
                if "정답" in question["explanation"]:
                    # 설명에서 정답 찾기
                    answer_parts = question["explanation"].split("정답")
                    if len(answer_parts) > 1:
                        answer_text = answer_parts[1].strip()
                        # 첫 문장 추출
                        first_sentence = re.search(r'^[^.!?]+[.!?]', answer_text)
                        if first_sentence:
                            fixed_question["question"] = first_sentence.group(0).strip()

                # 위 방법으로 추출 실패한 경우
                if fixed_question["question"] == question["question"]:
                    # "다음 중"으로 시작하는 질문 수정
                    if question["question"].startswith("다음 중"):
                        rest = question["question"].replace("다음 중", "").strip()
                        if "는" in rest or "은" in rest:
                            parts = re.split(r'(은|는)', rest, 1)
                            if len(parts) > 2:
                                subject = parts[0].strip()
                                verb = parts[1]  # 은/는
                                rest = parts[2].strip()
                                if rest.endswith("?"):
                                    rest = rest[:-1].strip()
                                fixed_question["question"] = f"{subject}{verb} {rest}."

                    # 여전히 수정되지 않은 경우 기본 형식으로 변환
                    if fixed_question["question"] == question["question"]:
                        # 기본값: 설명에 있는 내용 + 맞다/틀리다
                        if question["answer"]:
                            fixed_question["question"] = f"이 내용은 사실이다: '{question['explanation'].split('.')[0]}'."
                        else:
                            fixed_question["question"] = f"이 내용은 틀렸다: '{question['explanation'].split('.')[0]}'."

            fixed_questions.append(fixed_question)

        return fixed_questions


    def clean_ox_questions(self, ox_questions: List[Dict]) -> List[Dict]:
        """OX 문제 정리 및 형식 표준화"""
        cleaned_questions = []

        for question in ox_questions:
            cleaned_question = question.copy()

            # 질문에서 "question: " 접두사 제거
            if cleaned_question["question"]:
                cleaned_question["question"] = re.sub(r'^(question:|질문:)\s*', '',
                                                      cleaned_question["question"],
                                                      flags=re.IGNORECASE)

            # 설명에서 "explanation: " 접두사 및 불필요한 정보 제거
            if cleaned_question["explanation"]:
                # 참고 자료/링크 부분 제거
                if "```" in cleaned_question["explanation"]:
                    cleaned_question["explanation"] = cleaned_question["explanation"].split("```")[0].strip()

                # 설명에 있는 question/answer/explanation 태그 제거
                cleaned_question["explanation"] = re.sub(r'question:\s*[^\n]+', '',
                                                         cleaned_question["explanation"],
                                                         flags=re.IGNORECASE)
                cleaned_question["explanation"] = re.sub(r'answer:\s*[OXYNoxyn]', '',
                                                         cleaned_question["explanation"],
                                                         flags=re.IGNORECASE)
                cleaned_question["explanation"] = re.sub(r'explanation:\s*', '',
                                                         cleaned_question["explanation"],
                                                         flags=re.IGNORECASE)

                # 여러 줄 텍스트를 한 줄로 변환
                cleaned_question["explanation"] = ' '.join(cleaned_question["explanation"].split())

            # 질문이 객관식 형태인지 확인하고 수정
            if any(pattern in cleaned_question["question"].lower() for pattern in
                   ["다음 중", "무엇인가", "고르시오", "선택하시오", "선택하세요", "올바른 것은", "옳은 것은", "맞는 것은"]):

                # 진술문으로 변환 (기본값)
                statement = f"이 내용은 {'맞습니다' if cleaned_question['answer'] else '틀립니다'}: \"{cleaned_question['question']}\""

                # 설명에서 더 나은 진술문 찾기 시도
                if cleaned_question["explanation"]:
                    sentences = re.split(r'[.!?]\s+', cleaned_question["explanation"])
                    if sentences and len(sentences) > 0:
                        first_sentence = sentences[0].strip()
                        if len(first_sentence) > 10:  # 문장이 너무 짧지 않은지 확인
                            statement = first_sentence

                cleaned_question["question"] = statement

            # 빈 문제 방지
            if not cleaned_question["question"]:
                cleaned_question["question"] = "주어진 내용에 관한 OX 문제"

            if not cleaned_question["explanation"]:
                cleaned_question["explanation"] = f"해당 문제의 정답은 {'O' if cleaned_question['answer'] else 'X'}입니다."

            cleaned_questions.append(cleaned_question)

        return cleaned_questions

    def clean_multiple_choice(self, mc_questions: List[Dict]) -> List[Dict]:
        """객관식 문제 후처리 - 선택지 정리"""
        cleaned_questions = []

        for question in mc_questions:
            cleaned = question.copy()

            # A), B) 등이 포함된 선택지를 정리
            for i, option in enumerate(cleaned["options"]):
                # 선택지에서 A), 1) 등의 접두사 제거
                value = option["value"]
                value = re.sub(r'^[A-D1-4][\.:\)]\s*', '', value)
                cleaned["options"][i]["value"] = value.strip()

            # 설명에서 불필요한 텍스트 제거
            if cleaned["explanation"]:
                # "정답: A" 등의 패턴 제거
                cleaned["explanation"] = re.sub(r'정답\s*:?\s*[A-D1-4]', '', cleaned["explanation"])
                # 여러 줄을 한 줄로 합치기
                cleaned["explanation"] = ' '.join(cleaned["explanation"].split())

            cleaned_questions.append(cleaned)

        return cleaned_questions