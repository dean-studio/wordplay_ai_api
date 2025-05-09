from model_manager import ModelManager


class QuestionGenerator:
    """문제 생성 클래스"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def generate_multiple_choice(self, content: str, analysis: str, count: int = 2) -> str:
        """객관식 문제 생성"""
        system_message = f"""
- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.
- 당신은 교육 콘텐츠 기반 문제 생성 전문가이다.
- 주어진 교육 콘텐츠를 기반으로 객관식 문제 {count}개를 생성하라.
- 각 객관식 문제는 반드시 4개의 선택지를 포함해야 하며, 정답과 오답이 명확해야 한다.
- 모든 문제는 제공된 내용에만 기반해야 하며, 외부 지식을 요구하지 않아야 한다.
- 문제는 서로 중복되지 않고 다양한 내용을 다루어야 한다.
- 출력 형식:

## 객관식 문제 1
질문: [문제 내용]
보기1: [선택지1]
보기2: [선택지2]
보기3: [선택지3]
보기4: [선택지4]
정답: [정답 번호]
설명: [왜 이 답이 맞는지 설명]

## 객관식 문제 2
질문: [문제 내용]
보기1: [선택지1]
보기2: [선택지2]
보기3: [선택지3]
보기4: [선택지4]
정답: [정답 번호]
설명: [왜 이 답이 맞는지 설명]
"""

        user_message = f"다음 교육 콘텐츠와 분석 결과를 바탕으로 객관식 문제를 생성해주세요:\n\n# 교육 콘텐츠\n{content}\n\n# 분석 결과\n{analysis}"

        return self.model_manager.generate_text(
            system_message=system_message,
            user_message=user_message,
            max_new_tokens=2048,
            temperature=0.7
        )

    def generate_ox_questions(self, content: str, analysis: str, count: int = 2) -> str:
        """OX 문제 생성"""
        system_message = f"""
- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.
- 당신은 교육 콘텐츠 기반 문제 생성 전문가이다.
- 주어진 교육 콘텐츠를 기반으로 OX 문제 {count}개를 생성하라.
- 모든 문제는 제공된 내용에만 기반해야 하며, 외부 지식을 요구하지 않아야 한다.
- 출력 형식:

## OX 문제 1
질문: [문제 내용]
정답: [O 또는 X]
설명: [왜 O 또는 X인지 설명]

## OX 문제 2
질문: [문제 내용]
정답: [O 또는 X]
설명: [왜 O 또는 X인지 설명]
"""

        user_message = f"다음 교육 콘텐츠와 분석 결과를 바탕으로 OX 문제를 생성해주세요:\n\n# 교육 콘텐츠\n{content}\n\n# 분석 결과\n{analysis}"

        return self.model_manager.generate_text(
            system_message=system_message,
            user_message=user_message,
            max_new_tokens=1536,
            temperature=0.7
        )