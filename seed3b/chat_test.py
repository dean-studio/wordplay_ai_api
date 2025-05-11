import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# 허깅페이스에 공개된 CLOVA X 1.5B 모델 리포지토리 사용
model = AutoModelForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B")
tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B")

# GPU 사용 가능 시 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def chat_interface(user_input: str) -> str:
    # 클로바 X 샘플 템플릿: 시스템 메시지와 사용자 메시지 구성
    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 오늘은 2025년 04월 24일(목)이다.'},
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
    output_ids = model.generate(
        **inputs,
        max_length=5024,
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

with gr.Blocks(css=css) as demo:
    gr.Markdown("## CLOVA X Chat (1.5B)")
    chatbot = gr.Chatbot(label="CLOVA X Chat")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(placeholder="질문을 입력하세요.", show_label=False)
    txt.submit(respond, inputs=[txt, state], outputs=[txt, chatbot])

demo.launch(server_port=8283, server_name="0.0.0.0")