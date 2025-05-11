import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import time

# 모델 및 토크나이저 로드
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 양자화 설정 추가 (메모리 사용 최적화)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # bfloat16 대신 float16 사용
    device_map="auto",
    load_in_8bit=True,  # 8비트 양자화 사용
)


# 모델 예열을 위한 함수
def warm_up_model():
    print("모델 예열 중...")
    sample_input = "안녕하세요"
    messages = [
        {"role": "system", "content": "당신은 AI 어시스턴트입니다."},
        {"role": "user", "content": sample_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=20)
    print("모델 예열 완료")


# 처음 로딩 시 예열
warm_up_model()


def chat_interface(user_input: str) -> str:
    start_time = time.time()  # 시간 측정 시작

    # 짧은 시스템 프롬프트 사용
    system_prompt = "당신은 유능한 AI 어시스턴트입니다."

    # 간단한 요약 감지
    if "요약" in user_input:
        system_prompt += " 다음 텍스트를 자세하게 요약해주세요."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # 채팅 템플릿 적용
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 입력 토큰화
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 토큰화된 입력의 길이 확인
    input_length = inputs.input_ids.shape[1]

    # 생성 매개변수 최적화
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,  # 짧게 제한
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )

    # 생성된, 입력을 제외한 텍스트만 디코딩
    generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

    elapsed_time = time.time() - start_time
    print(f"응답 생성 시간: {elapsed_time:.2f}초")

    return generated_text


def respond(message, chat_history):
    try:
        response = chat_interface(message)
        chat_history.append((message, response))
    except Exception as e:
        chat_history.append((message, f"오류 발생: {str(e)}"))
    return "", chat_history


# Gradio 인터페이스 설정
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

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Bllossom-8B 한국어 챗봇")
    chatbot = gr.Chatbot(label="Bllossom Chat")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(placeholder="메시지를 입력하세요", show_label=False)
    txt.submit(respond, inputs=[txt, state], outputs=[txt, chatbot])

demo.queue().launch(server_port=8283, server_name="0.0.0.0")