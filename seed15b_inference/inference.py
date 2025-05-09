import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr
import os
import random

# 원본 모델 및 토크나이저 로드
base_model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# LoRA 모델 경로 설정
lora_paths = [
    "../seed15b_lora/clova-lora-qa-final",
    "../seed15b_lora_wp/wpdb-lora-tuned-final"
]

# 모델 로드
print("기본 모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 어댑터 적용
for lora_path in lora_paths:
    if os.path.exists(lora_path):
        print(f"LoRA 어댑터 로드 중: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"LoRA 어댑터 {os.path.basename(lora_path)}가 성공적으로 적용되었습니다.")
    else:
        print(f"경고: LoRA 어댑터 경로({lora_path})를 찾을 수 없습니다.")

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")

# 카테고리 매핑 (내용 분석에 따라 자동 선택)
CATEGORIES = {
    "경제": [101, "경제", "경제 개념"],
    "과학": [201, "과학", "자연과학"],
    "사회": [301, "사회", "사회학"],
    "국어": [401, "국어", "언어"],
    "수학": [501, "수학", "수리"],
    "역사": [601, "역사", "한국사"],
    "영어": [701, "영어", "외국어"],
    "기타": [901, "기타", "일반상식"]
}

# 학년 매핑
GRADE_LEVELS = [
    "초등 1학년", "초등 2학년", "초등 3학년", "초등 4학년", "초등 5학년", "초등 6학년",
    "중등 1학년", "중등 2학년", "중등 3학년",
    "고등 1학년", "고등 2학년", "고등 3학년"
]


# 첫 단계: 모델에게 교육 내용 분석 요청 (카테고리 및 핵심 개념 추출)
def analyze_content(content):
    system_message = """
- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.
- 당신은 교육 콘텐츠 분석 전문가이다.
- 제공된 텍스트를 분석하여 다음 정보를 정확히 추출하라:
  1. 교과 카테고리 (경제, 과학, 사회, 국어, 수학, 역사, 영어, 기타 중 하나)
  2. 적절한 학년 수준 (초등 1-6학년, 중등 1-3학년, 고등 1-3학년 중 하나)
  3. 난이도 (1-쉬움, 2-보통, 3-어려움)
  4. 해당 내용의 핵심 개념 키워드 5개 (쉼표로 구분)
  5. 내용 요약 (1-2문장)

- 출력 형식:
카테고리: [카테고리명]
학년: [학년 수준]
난이도: [1-3]
핵심개념: [키워드1], [키워드2], [키워드3], [키워드4], [키워드5]
요약: [내용 요약]
"""

    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"다음 교육 콘텐츠를 분석해주세요:\n\n{content}"}
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=1024,
            temperature=0.3,
            stop_strings=["<|endofturn|>", "<|stop|>"],
            tokenizer=tokenizer
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    full_output = "\n".join(output)

    if "assistant" in full_output:
        analysis = full_output.split("assistant", 1)[1].strip()
        return analysis
    else:
        return full_output


# 두 번째 단계: 모델에게 문제 생성 요청 (구조화되지 않은 형태)
def generate_questions(content, analysis):
    system_message = """
- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.
- 당신은 교육 콘텐츠 기반 문제 생성 전문가이다.
- 주어진 교육 콘텐츠를 기반으로 객관식 문제 1개와 OX 문제 1개를 생성하라.
- 객관식 문제는 반드시 4개의 선택지를 포함해야 하며, 정답과 오답이 명확해야 한다.
- 모든 문제는 제공된 내용에만 기반해야 하며, 외부 지식을 요구하지 않아야 한다.
- 출력 형식:

## 객관식 문제
질문: [문제 내용]
보기1: [선택지1]
보기2: [선택지2]
보기3: [선택지3]
보기4: [선택지4]
정답: [정답 번호]
설명: [왜 이 답이 맞는지 설명]

## OX 문제
질문: [문제 내용]
정답: [O 또는 X]
설명: [왜 O 또는 X인지 설명]
"""

    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"다음 교육 콘텐츠와 분석 결과를 바탕으로 문제를 생성해주세요:\n\n# 교육 콘텐츠\n{content}\n\n# 분석 결과\n{analysis}"}
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            stop_strings=["<|endofturn|>", "<|stop|>"],
            tokenizer=tokenizer
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    full_output = "\n".join(output)

    if "assistant" in full_output:
        questions = full_output.split("assistant", 1)[1].strip()
        return questions
    else:
        return full_output


# 세 번째 단계: 생성된 문제를 JSON으로 구조화
def parse_questions_to_json(questions, analysis, content):
    result = {}

    # 분석 정보 파싱
    category = "기타"
    grade_level = "초등 5학년"
    difficulty = 2
    summary = ""

    category_match = re.search(r"카테고리:\s*(\S+)", analysis)
    grade_match = re.search(r"학년:\s*([^\n]+)", analysis)
    difficulty_match = re.search(r"난이도:\s*(\d)", analysis)
    summary_match = re.search(r"요약:\s*([^\n]+)", analysis)

    if category_match:
        category = category_match.group(1)
    if grade_match:
        grade_level = grade_match.group(1).strip()
    if difficulty_match:
        difficulty = int(difficulty_match.group(1))
    if summary_match:
        summary = summary_match.group(1).strip()

    # 카테고리 매핑에서 인덱스 찾기
    category_idx = 901  # 기본값
    for key, value in CATEGORIES.items():
        if key in category or category in key:
            category_idx = value[0]
            category = value[1]
            break

    # 기본 결과 구조 설정
    result = {
        "question": f"{category} - {summary}",
        "category": category,
        "category_idx": category_idx,
        "grade_level": grade_level if grade_level in GRADE_LEVELS else "초등 5학년",
        "difficulty": difficulty if 1 <= difficulty <= 3 else 2,
        "multiple_choice": {
            "question": "",
            "options": [
                {"id": 1, "value": ""},
                {"id": 2, "value": ""},
                {"id": 3, "value": ""},
                {"id": 4, "value": ""}
            ],
            "answer": 1,
            "explanation": ""
        },
        "ox_quiz": {
            "question": "",
            "answer": True,
            "explanation": ""
        },
        "reference_links": {
            "priority_1": f"https://namu.wiki/w/{category}",
            "priority_2": f"https://blog.naver.com/search/blog.naver?query={category}",
            "priority_3": f"https://ko.wikipedia.org/wiki/{category}"
        }
    }

    # 객관식 문제 파싱
    mc_section = re.search(r"## 객관식 문제(.*?)(?=## OX 문제|\Z)", questions, re.DOTALL)
    if mc_section:
        mc_text = mc_section.group(1)

        question_match = re.search(r"질문:\s*([^\n]+)", mc_text)
        if question_match:
            result["multiple_choice"]["question"] = question_match.group(1).strip()

        options = re.findall(r"보기(\d):\s*([^\n]+)", mc_text)
        for opt in options:
            idx = int(opt[0])
            if 1 <= idx <= 4:
                result["multiple_choice"]["options"][idx - 1]["value"] = opt[1].strip()

        answer_match = re.search(r"정답:\s*(\d)", mc_text)
        if answer_match:
            result["multiple_choice"]["answer"] = int(answer_match.group(1))

        explanation_match = re.search(r"설명:\s*([^\n]+)", mc_text)
        if explanation_match:
            result["multiple_choice"]["explanation"] = explanation_match.group(1).strip()

    # OX 문제 파싱
    ox_section = re.search(r"## OX 문제(.*?)(?=##|\Z)", questions, re.DOTALL)
    if ox_section:
        ox_text = ox_section.group(1)

        question_match = re.search(r"질문:\s*([^\n]+)", ox_text)
        if question_match:
            result["ox_quiz"]["question"] = question_match.group(1).strip()

        answer_match = re.search(r"정답:\s*([OXox])", ox_text)
        if answer_match:
            answer_text = answer_match.group(1).upper()
            result["ox_quiz"]["answer"] = True if answer_text == "O" else False

        explanation_match = re.search(r"설명:\s*([^\n]+)", ox_text)
        if explanation_match:
            result["ox_quiz"]["explanation"] = explanation_match.group(1).strip()

    # 오류 검사 및 수정 (필수 필드가 비어있는 경우)
    if not result["multiple_choice"]["question"]:
        result["multiple_choice"]["question"] = f"{category}에 관한 문제"

    if not result["ox_quiz"]["question"]:
        result["ox_quiz"]["question"] = f"{category}에 관한 진위형 문제"

    # 빈 옵션 처리
    for i in range(4):
        if not result["multiple_choice"]["options"][i]["value"]:
            result["multiple_choice"]["options"][i]["value"] = f"선택지 {i + 1}"

    return result


# 전체 파이프라인 연결
def generate_quiz(content):
    try:
        # 1. 내용 분석
        analysis = analyze_content(content)
        print("=== 분석 결과 ===")
        print(analysis)

        # 2. 문제 생성
        questions = generate_questions(content, analysis)
        print("=== 생성된 문제 ===")
        print(questions)

        # 3. JSON 파싱 및 구조화
        result_json = parse_questions_to_json(questions, analysis, content)
        print("=== 구조화된 JSON ===")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))

        # 4. 최종 결과 반환
        return json.dumps(result_json, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        # 오류 발생 시 기본 JSON 반환
        return json.dumps({
            "error": str(e),
            "question": "오류가 발생했습니다",
            "category": "기타",
            "category_idx": 999,
            "grade_level": "초등 5학년",
            "difficulty": 2,
            "multiple_choice": {
                "question": "오류로 인해 문제를 생성할 수 없습니다",
                "options": [
                    {"id": 1, "value": "오류 발생"},
                    {"id": 2, "value": "다시 시도해주세요"},
                    {"id": 3, "value": "입력 내용을 확인해주세요"},
                    {"id": 4, "value": "더 짧은 내용으로 시도해보세요"}
                ],
                "answer": 1,
                "explanation": "시스템 오류가 발생했습니다. 다시 시도해주세요."
            },
            "ox_quiz": {
                "question": "이 결과는 오류 때문에 생성된 것입니다",
                "answer": True,
                "explanation": "오류가 발생하여 기본 응답을 반환했습니다."
            },
            "reference_links": {
                "priority_1": "",
                "priority_2": "",
                "priority_3": ""
            }
        }, ensure_ascii=False, indent=2)


# 최종 채팅 인터페이스 구현
def chat_interface(user_input):
    return generate_quiz(user_input)


def respond(message, chat_history):
    response = chat_interface(message)
    chat_history.append((message, response))
    return "", chat_history


# Gradio UI 설정
css = """
html, body {
    width: 100%;
    height: 100%;
    margin: 0;
    padding: 0;
}
.gradio-container {
    width: 100% !important;
    height: 100% !important;
    margin: 0;
    padding: 0;
}
pre {
    white-space: pre-wrap;
    overflow-x: auto;
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## 교육 콘텐츠 문제 생성기 (하이브리드 방식)")

    with gr.Row():
        with gr.Column(scale=6):
            txt = gr.Textbox(
                placeholder="교과서나 학습 자료의 내용을 붙여넣으면 자동으로 문제가 생성됩니다.",
                show_label=False,
                lines=10
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("문제 생성하기")

    output_json = gr.Code(language="json", label="생성된 JSON 문제")


    submit_btn.click(
        fn=lambda x: chat_interface(x),
        inputs=[txt],
        outputs=[output_json]
    )

# 서버 시작
print("Gradio 서버 시작 중...")
demo.launch(server_port=7860, server_name="0.0.0.0")