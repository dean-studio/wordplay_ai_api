import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

model = AutoModelForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B")
tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B")

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

    try:
        assistant_response = full_output.split("assistant")[-1].strip()
        return assistant_response
    except:
        return full_output


def respond(message, chat_history):
    response = chat_interface(message)
    chat_history.append((message, response))
    return "", chat_history


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