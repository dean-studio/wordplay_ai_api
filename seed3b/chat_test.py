import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import re

# 모델 및 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
                                          trust_remote_code=True)

# GPU 사용 가능 시 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def chat_interface(user_input: str) -> str:
    # 요약 요청인지 감지
    is_summary_request = any(keyword in user_input.lower() for keyword in ["요약", "정리", "summarize", "summary"])

    # 요약 요청인 경우 프롬프트 강화
    if is_summary_request:
        # 요약을 위한 명시적 지시사항 추가
        system_content = '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 오늘은 2025년 04월 24일(목)이다.\n- 요약 요청 시 핵심 내용, 인물, 배경, 분위기를 포함하여 500자 이상으로 상세히 요약한다.'

        # 요약 요청 패턴 감지 (텍스트 본문 추출)
        text_to_summarize = re.sub(r'요약.*?[:：]', '', user_input).strip()
        if text_to_summarize == user_input:  # 패턴이 없으면 원본 사용
            text_to_summarize = user_input

        # 강화된 요약 요청 형식
        enhanced_input = f"""다음 텍스트를 매우 상세하게 요약해주세요.
1. 최소 500자 이상으로 작성할 것
2. 주요 내용, 등장인물, 배경, 분위기를 모두 포함할 것
3. 간단하게 한 줄로 요약하지 말고 상세하게 요약할 것
4. 텍스트에 나온 중요한 정보를 놓치지 말 것

텍스트:
{text_to_summarize}"""

    else:
        system_content = '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 오늘은 2025년 04월 24일(목)이다.'
        enhanced_input = user_input

    # Few-shot 예제 추가 (요약 요청인 경우만)
    if is_summary_request:
        chat = [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_content},
            {"role": "user", "content": "짧은 소설을 요약해줘"},
            {"role": "assistant", "content": "요약할 소설을 제공해주시면 등장인물, 배경, 주요 사건, 분위기를 포함하여 500자 이상으로 상세히 요약해드리겠습니다."},
            {"role": "user", "content": enhanced_input}
        ]
    else:
        chat = [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_content},
            {"role": "user", "content": enhanced_input}
        ]

    # 채팅 템플릿을 적용해 모델 입력 데이터 생성
    inputs = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 모델을 통한 응답 생성 (매개변수 조정)
    output_ids = model.generate(
        **inputs,
        max_length=5024,
        min_length=300,  # 최소 길이 설정 (요약의 경우)
        do_sample=True,
        temperature=0.8,  # 약간의 창의성 부여
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,  # 반복 방지
        stop_strings=["<|endofturn|>", "<|stop|>"],
        tokenizer=tokenizer
    )

    # 디코딩 및 응답 추출
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    full_output = "\n".join(output)

    # assistant 태그 이후의 내용만 추출
    try:
        # 마지막 assistant 태그 이후의 내용 추출
        assistant_response = full_output.split("assistant")[-1].strip()

        # 응답이 너무 짧으면 (50자 미만) 다시 시도
        if is_summary_request and len(assistant_response) < 50:
            print("응답이 너무 짧아 다시 시도합니다.")
            # 다른 접근 방식으로 다시 시도
            chat = [
                {"role": "tool_list", "content": ""},
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"다음 텍스트의 상세한 요약을 500자 이상으로 작성해주세요: {text_to_summarize}"}
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
                min_length=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                stop_strings=["<|endofturn|>", "<|stop|>"],
                tokenizer=tokenizer
            )

            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            full_output = "\n".join(output)
            assistant_response = full_output.split("assistant")[-1].strip()

        return assistant_response
    except:
        return full_output


def respond(message, chat_history):
    response = chat_interface(message)
    chat_history.append((message, response))
    return "", chat_history


# Custom CSS: 브라우저 전체 화면으로 조정
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
    gr.Markdown("## CLOVA X Chat (3B)")
    chatbot = gr.Chatbot(label="CLOVA X Chat")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(placeholder="질문을 입력하세요.", show_label=False)
    txt.submit(respond, inputs=[txt, state], outputs=[txt, chatbot])

demo.launch(server_port=8283, server_name="0.0.0.0")