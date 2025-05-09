import json
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
                "../seed15b_lora/clova-lora-qa-final",
                "../seed15b_lora_wp/wpdb-lora-tuned-final"
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

            # 4. 문제 파싱
            mc_questions = self.parser.parse_multiple_choice(mc_questions_text)
            print("=== 파싱된 객관식 문제 ===")
            print(json.dumps(mc_questions, ensure_ascii=False, indent=2))

            ox_questions = self.parser.parse_ox_questions(ox_questions_text)
            print("=== 파싱된 OX 문제 ===")
            print(json.dumps(ox_questions, ensure_ascii=False, indent=2))

            # 5. 파싱된 문제 후처리 - 상대적 시간 표현 변환
            mc_questions = self.post_process_questions(mc_questions, current_date)
            ox_questions = self.post_process_questions(ox_questions, current_date)

            # 6. 최종 결과 구성
            result = []

            # 객관식 문제 추가 (최대 허용 개수까지)
            for i, mc in enumerate(mc_questions[:mc_count]):
                result.append({
                    "multiple_choice": mc
                })

            # 객관식 문제가 없거나 부족하면 기본 문제 추가
            while len(result) < mc_count:
                result.append({
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
                result.append({
                    "ox_quiz": ox
                })

            # OX 문제가 없거나 부족하면 기본 문제 추가
            while len(result) - mc_count < ox_count:
                result.append({
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

        result = []

        # 기본 객관식 문제 추가
        for i in range(mc_count):
            result.append({
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
            result.append({
                "ox_quiz": {
                    "question": f"OX 퀴즈 질문 {i + 1}",
                    "answer": i % 2 == 0,  # 홀수 번째는 True, 짝수 번째는 False
                    "explanation": f"정답은 {'O' if i % 2 == 0 else 'X'}입니다. 이유: 시스템 오류로 기본 응답이 생성되었습니다. (현재 날짜: {date_info})"
                }
            })

        return json.dumps(result, ensure_ascii=False, indent=2)