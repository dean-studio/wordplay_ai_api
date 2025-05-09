from model_manager import ModelManager
from datetime import datetime


class ContentAnalyzer:
    """교육 콘텐츠 분석 클래스"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def analyze(self, content: str) -> str:
        """콘텐츠 분석 실행"""
        # 현재 날짜 정보 추가
        current_date = datetime.now()
        date_info = f"현재 날짜는 {current_date.year}년 {current_date.month}월 {current_date.day}일입니다."

        system_message = f"""
- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.
- {date_info}
- 당신은 교육 콘텐츠 분석 전문가이다.
- 제공된 텍스트를 분석하여 다음 정보를 정확히 추출하라:
  1. 교과 카테고리 (경제, 과학, 사회, 국어, 수학, 역사, 영어, 기타 중 하나)
  2. 핵심 개념 3개 (쉼표로 구분)
  3. 글이 작성된 연도나 시기에 대한 추정 (가능한 경우)

- 출력 형식:
카테고리: [카테고리명]
핵심개념: [키워드1], [키워드2], [키워드3]
시기정보: [글이 작성된 것으로 추정되는 연도/시기 또는 '확인 불가']
"""
        user_message = f"다음 교육 콘텐츠를 분석해주세요:\n\n{content}"

        return self.model_manager.generate_text(
            system_message=system_message,
            user_message=user_message,
            max_new_tokens=1024,
            temperature=0.3
        )