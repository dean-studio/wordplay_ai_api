import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# 모델 및 토크나이저 로드
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # 또는 torch.float16
    device_map="auto"
)

# GPU 사용 가능 시 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def chat_interface(user_input: str) -> str:
    # 요약 요청인지 감지
    is_summary_request = any(keyword in user_input.lower() for keyword in ["요약", "정리", "summarize", "summary"])

    if is_summary_request:
        # 요약할 텍스트 추출 (요약: 이후의 내용)
        text_to_summarize = user_input
        if ":" in user_input:
            text_to_summarize = user_input.split(":", 1)[1].strip()

        # 요약을 위한 명시적 프롬프트 작성
        system_prompt = "당신은 유능한 AI 어시스턴트입니다. 다음 텍스트를 자세하게 요약해주세요. 핵심 내용, 등장인물, 배경, 분위기를 모두 포함하여 500자 이상으로 상세히 요약하세요."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 텍스트를 요약해주세요:\n\n{text_to_summarize}"}
        ]
    else:
        # 일반 대화
        system_prompt = "당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 대해 친절하게 답변해주세요."
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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 토큰화된 입력의 길이 확인
    input_length = inputs.input_ids.shape[1]

    # 종료 토큰 설정
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else None
    ]
    terminators = [t for t in terminators if t is not None]

    # 모델 생성 매개변수 설정 (요약 요청인 경우 더 긴 출력)
    max_new_tokens = 2048 if is_summary_request else 1024

    # 생성 수행
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=300 if is_summary_request else 100,  # 요약의 경우 최소 길이 설정
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=terminators,
        )

    # 생성된, 입력을 제외한 텍스트만 디코딩
    generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

    return generated_text


def respond(message, chat_history):
    response = chat_interface(message)
    chat_history.append((message, response))
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
    gr.Markdown("## Bllossom-8B 한국어 요약 챗봇")
    chatbot = gr.Chatbot(label="Bllossom Chat")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(placeholder="텍스트를 입력하거나 '요약: [텍스트]' 형식으로 요약을 요청하세요.", show_label=False)
    txt.submit(respond, inputs=[txt, state], outputs=[txt, chatbot])

demo.launch(server_port=8283, server_name="0.0.0.0")