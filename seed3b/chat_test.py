import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

model = AutoModelForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B")
tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def chat_interface(user_input: str) -> str:
    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 오늘은 2025년 04월 24일(목)이다.'},
        {"role": "user", "content": user_input}
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_length=5024,
        stop_strings=["<|endofturn|>", "<|stop|>"],
        tokenizer=tokenizer
    )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    full_output = "\n".join(output)

    if "assistant" in full_output:
        assistant_reply = full_output.split("assistant", 1)[1].strip()
        return assistant_reply
    else:
        return full_output


demo = gr.ChatInterface(
    fn=chat_interface,
    title="CLOVA X 3B 챗봇",
    description="네이버의 HyperCLOVAX 3B 모델을 이용한 한국어 챗봇입니다",
    theme=gr.themes.Soft(),
    examples=[
        "안녕하세요, 자기소개 부탁해요.",
        "한국의 유명한 관광지 추천해주세요.",
        "인공지능에 대해 간단히 설명해줄래요?"
    ],
    retry_btn="다시 생성",
    undo_btn="되돌리기",
    clear_btn="대화 초기화"
)

demo.launch(server_port=8283, server_name="0.0.0.0", share=True)