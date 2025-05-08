import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr
import os

# 원본 모델 및 토크나이저 로드
base_model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# LoRA 모델 경로 설정 (이 경로를 실제 저장된 LoRA 어댑터 경로로 변경)
# 예: ../seed15b_lora/clova-lora-qa-final 또는 ../seed15b_lora/clova-lora-korquad-only
lora_path = "../seed15b_lora/clova-lora-qa-final"  # 실제 경로로 변경하세요

# 모델 로드
print("기본 모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 어댑터 적용
if os.path.exists(lora_path):
    print(f"LoRA 어댑터 로드 중: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    print("LoRA 어댑터가 성공적으로 적용되었습니다.")
else:
    print(f"경고: LoRA 어댑터 경로({lora_path})를 찾을 수 없습니다. 기본 모델을 사용합니다.")

# 모델을 평가 모드로 설정
model.eval()

# GPU 사용 가능 시 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")


def chat_interface(user_input: str) -> str:
    # 클로바 X 샘플 템플릿: 시스템 메시지와 사용자 메시지 구성
    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system",
         "content": '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 학습은 LoRA 기법으로 QA 특화 학습이 수행되었다.\n- 오늘은 2025년 05월 08일(목)이다.'},
        {"role": "user", "content": user_input}
    ]

    # 채팅 템플릿을 적용해 모델 입력 데이터 생성 (generation prompt 포함)
    inputs = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 모델을 통한 응답 생성
    with torch.no_grad():  # 추론 시 그라디언트 계산 방지
        output_ids = model.generate(
            **inputs,
            max_length=2048,  # 더 짧은 길이로 제한
            stop_strings=["<|endofturn|>", "<|stop|>"],
            tokenizer=tokenizer
        )

    # 디코딩: list 형태인 결과를 join하여 전체 텍스트 생성
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    full_output = "\n".join(output)

    # "assistant" 태그 이후의 내용만 추출 (태그가 없으면 전체 텍스트 반환)
    if "assistant" in full_output:
        assistant_reply = full_output.split("assistant", 1)[1].strip()
        return assistant_reply
    else:
        return full_output


def respond(message, chat_history):
    response = chat_interface(message)
    chat_history.append((message, response))
    return "", chat_history


# Custom CSS: 브라우저 전체 화면(100% width/height)으로 조정
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
"""

# Gradio 인터페이스 설정
with gr.Blocks(css=css) as demo:
    gr.Markdown("## CLOVA X Chat (1.5B) - LoRA QA 특화 모델")
    chatbot = gr.Chatbot(label="CLOVA X Chat")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(placeholder="질문을 입력하세요. 특히 QA 성능이 향상되었습니다.", show_label=False)

    # 예제 질문 추가
    gr.Examples([
        ["다음 문서를 읽고 질문에 답하세요. 문서: 대한민국의 수도는 서울이며, 부산은 제2의 도시이다. 서울은 한강이 흐르며 인구가 약 1000만 명이다. 질문: 대한민국의 수도는 어디인가?"],
        ["인공지능에 대해 설명해줄래?"],
        ["심리학에서 인지부조화란 무엇인가요?"]
    ], txt)

    txt.submit(respond, inputs=[txt, state], outputs=[txt, chatbot])

# 서버 시작
print("Gradio 서버 시작 중...")
demo.launch(server_port=8283, server_name="0.0.0.0")