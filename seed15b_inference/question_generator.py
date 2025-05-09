from model_manager import ModelManager


class QuestionGenerator:
    """문제 생성 클래스"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def generate_multiple_choice(self, content: str, analysis: str, count: int = 2) -> str:
        """객관식 문제 생성 - 개선된 지시사항"""
        system_message = f"""
    - AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.
    - 당신은 교육 콘텐츠 기반 문제 생성 전문가이다.
    - 주어진 교육 콘텐츠를 기반으로 객관식 문제 {count}개를 생성하라.
    - 각 객관식 문제는 반드시 4개의 선택지를 포함해야 하며, 정답과 오답이 명확해야 한다.
    - 모든 문제는 제공된 내용에만 기반해야 하며, 외부 지식을 요구하지 않아야 한다.
    - 문제는 서로 중복되지 않고 다양한 내용을 다루어야 한다.
    - 중요한 지침:
      1. 상대적 시간 표현("작년", "최근", "지난 달" 등) 대신 구체적인 연도나 날짜를 사용하라.
      2. 모든 정보는 사실에 기반하며, 구체적이고 정확한 수치와 데이터를 포함하라.
      3. 애매한 표현이나 일반화는 피하고, 명확하고 검증 가능한 내용을 사용하라.
      4. 설명은 간결하고 핵심적인 내용만 포함하라 (50단어 이내).
      5. 내용 출처가 있다면 "~에 따르면" 형식으로 포함하라.

    - 출력 형식:

    ## 객관식 문제 1
    질문: [명확하고 구체적인 문제 내용]
    보기1: [선택지1]
    보기2: [선택지2]
    보기3: [선택지3]
    보기4: [선택지4]
    정답: [정답 번호]
    설명: [간결하고 명확한 설명, 가능하면 구체적인 수치나 사실 포함]

    ## 객관식 문제 2
    질문: [명확하고 구체적인 문제 내용]
    보기1: [선택지1]
    보기2: [선택지2]
    보기3: [선택지3]
    보기4: [선택지4]
    정답: [정답 번호]
    설명: [간결하고 명확한 설명, 가능하면 구체적인 수치나 사실 포함]
    """

        # 나머지 코드는 동일...

        user_message = f"다음 교육 콘텐츠와 분석 결과를 바탕으로 객관식 문제를 생성해주세요:\n\n# 교육 콘텐츠\n{content}\n\n# 분석 결과\n{analysis}"

        return self.model_manager.generate_text(
            system_message=system_message,
            user_message=user_message,
            max_new_tokens=2048,
            temperature=0.7
        )

    def generate_ox_questions(self, content: str, analysis: str, count: int = 2) -> str:
        """OX 문제 생성 - 개선된 지시사항"""
        system_message = f"""
    - AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.
    - 당신은 교육 콘텐츠 기반 문제 생성 전문가이다.
    - 주어진 교육 콘텐츠를 기반으로 OX 문제 {count}개를 생성하라.
    - 모든 문제는 제공된 내용에만 기반해야 하며, 외부 지식을 요구하지 않아야 한다.
    - 중요한 지침:
      1. 상대적 시간 표현("작년", "최근", "지난 달" 등) 대신 구체적인 연도나 날짜를 사용하라.
      2. 모든 정보는 사실에 기반하며, 구체적이고 정확한 수치와 데이터를 포함하라.
      3. 애매한 표현이나 일반화는 피하고, 명확하고 검증 가능한 내용을 사용하라.
      4. 설명은 간결하고 핵심적인 내용만 포함하라 (50단어 이내).
      5. 내용 출처가 있다면 "~에 따르면" 형식으로 포함하라.

    - 출력 형식:

    ## OX 문제 1
    질문: [명확하고 구체적인 문제 내용]
    정답: [O 또는 X]
    설명: [간결하고 명확한 설명, 가능하면 구체적인 수치나 사실 포함]

    ## OX 문제 2
    질문: [명확하고 구체적인 문제 내용]
    정답: [O 또는 X]
    설명: [간결하고 명확한 설명, 가능하면 구체적인 수치나 사실 포함]
    """

        # 나머지 코드는 동일...

        user_message = f"다음 교육 콘텐츠와 분석 결과를 바탕으로 OX 문제를 생성해주세요:\n\n# 교육 콘텐츠\n{content}\n\n# 분석 결과\n{analysis}"

        return self.model_manager.generate_text(
            system_message=system_message,
            user_message=user_message,
            max_new_tokens=1536,
            temperature=0.7
        )