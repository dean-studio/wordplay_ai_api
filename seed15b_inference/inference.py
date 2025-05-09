import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr
import os

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


# 첫 단계: 간단한 내용 분석 요청 (카테고리 파악)
def analyze_content(content):
    system_message = """
- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.
- 당신은 교육 콘텐츠 분석 전문가이다.
- 제공된 텍스트를 분석하여 다음 정보를 정확히 추출하라:
  1. 교과 카테고리 (경제, 과학, 사회, 국어, 수학, 역사, 영어, 기타 중 하나)
  2. 핵심 개념 3개 (쉼표로 구분)

- 출력 형식:
카테고리: [카테고리명]
핵심개념: [키워드1], [키워드2], [키워드3]
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
            max_length=1024,  # max_length 증가
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


# 두 번째 단계: 모델에게 객관식 문제 생성 요청
def generate_multiple_choice_questions(content, analysis, count=2):
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

    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_message},
        {"role": "user",
         "content": f"다음 교육 콘텐츠와 분석 결과를 바탕으로 객관식 문제를 생성해주세요:\n\n# 교육 콘텐츠\n{content}\n\n# 분석 결과\n{analysis}"}
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


# 세 번째 단계: 모델에게 OX 문제 생성 요청
def generate_ox_questions(content, analysis, count=2):  # 기본값 2로 변경
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

    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_message},
        {"role": "user",
         "content": f"다음 교육 콘텐츠와 분석 결과를 바탕으로 OX 문제를 생성해주세요:\n\n# 교육 콘텐츠\n{content}\n\n# 분석 결과\n{analysis}"}
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
            max_length=1536,  # max_length 증가
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


# 객관식 문제 파싱 함수
def parse_multiple_choice(text):
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


# OX 문제 파싱 함수
def parse_ox_questions(text):
    ox_questions = []

    # 각 OX 문제 블록 찾기
    ox_blocks = re.findall(r"## OX 문제 \d+(.*?)(?=## OX 문제 \d+|$)", text, re.DOTALL)

    for block in ox_blocks:
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

        answer_match = re.search(r"정답:\s*([OXox])", block)
        if answer_match:
            answer_text = answer_match.group(1).upper()
            ox_obj["answer"] = True if answer_text == "O" else False

        explanation_match = re.search(r"설명:\s*([^\n]+)", block)
        if explanation_match:
            ox_obj["explanation"] = explanation_match.group(1).strip()

        ox_questions.append(ox_obj)

    return ox_questions


# 통합 파이프라인
def generate_quiz(content, mc_count=2, ox_count=2):  # 기본값 업데이트
    try:
        # 1. 내용 분석
        analysis = analyze_content(content)
        print("=== 분석 결과 ===")
        print(analysis)

        # 2. 객관식 문제 생성
        mc_questions_text = generate_multiple_choice_questions(content, analysis, mc_count)
        print("=== 생성된 객관식 문제 ===")
        print(mc_questions_text)

        # 3. OX 문제 생성
        ox_questions_text = generate_ox_questions(content, analysis, ox_count)
        print("=== 생성된 OX 문제 ===")
        print(ox_questions_text)

        # 4. 문제 파싱
        mc_questions = parse_multiple_choice(mc_questions_text)
        ox_questions = parse_ox_questions(ox_questions_text)

        # 5. 최종 결과 구성
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
                    "explanation": "정답은 '보기1'입니다. 이유: 기본 설명입니다."
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
                    "explanation": "정답은 O입니다. 이유: 기본 설명입니다."
                }
            })

        print("=== 최종 JSON ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        # 오류 발생 시 기본 JSON 반환
        return json.dumps([
            {
                "multiple_choice": {
                    "question": "객관식 문제 1",
                    "options": [
                        {"id": 1, "value": "보기1"},
                        {"id": 2, "value": "보기2"},
                        {"id": 3, "value": "보기3"},
                        {"id": 4, "value": "보기4"}
                    ],
                    "answer": 1,
                    "explanation": "정답은 '보기1'입니다. 이유: 시스템 오류로 기본 응답이 생성되었습니다."
                }
            },
            {
                "multiple_choice": {
                    "question": "객관식 문제 2",
                    "options": [
                        {"id": 1, "value": "보기1"},
                        {"id": 2, "value": "보기2"},
                        {"id": 3, "value": "보기3"},
                        {"id": 4, "value": "보기4"}
                    ],
                    "answer": 1,
                    "explanation": "정답은 '보기1'입니다. 이유: 시스템 오류로 기본 응답이 생성되었습니다."
                }
            },
            {
                "ox_quiz": {
                    "question": "OX 퀴즈 질문 1",
                    "answer": True,
                    "explanation": "정답은 O입니다. 이유: 시스템 오류로 기본 응답이 생성되었습니다."
                }
            },
            {
                "ox_quiz": {
                    "question": "OX 퀴즈 질문 2",
                    "answer": False,
                    "explanation": "정답은 X입니다. 이유: 시스템 오류로 기본 응답이 생성되었습니다."
                }
            }
        ], ensure_ascii=False, indent=2)


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
    gr.Markdown("## 교육 콘텐츠 문제 생성기 (하이브리드 접근법)")

    with gr.Row():
        with gr.Column(scale=6):
            txt = gr.Textbox(
                placeholder="교과서나 학습 자료의 내용을 붙여넣으면 자동으로 문제가 생성됩니다.",
                show_label=False,
                lines=10
            )
        with gr.Column(scale=1):
            with gr.Row():
                mc_slider = gr.Slider(minimum=1, maximum=2, value=2, step=1, label="객관식 문제 수")
                ox_slider = gr.Slider(minimum=1, maximum=2, value=2, step=1, label="OX 문제 수")  # 기본값 2로 변경, 최대값도 2로 변경
            submit_btn = gr.Button("문제 생성하기")

    output_json = gr.Code(language="json", label="생성된 JSON 문제")

    submit_btn.click(
        fn=lambda x, mc, ox: generate_quiz(x, int(mc), int(ox)),
        inputs=[txt, mc_slider, ox_slider],
        outputs=[output_json]
    )

# 서버 시작
print("Gradio 서버 시작 중...")
demo.launch(server_port=7860, server_name="0.0.0.0")