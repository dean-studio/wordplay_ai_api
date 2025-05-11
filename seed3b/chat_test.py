import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import gradio as gr

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

# 모델, 프로세서, 토크나이저 로드
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"모델을 {device}에 로드했습니다.")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")


def process_image_and_text(image, user_input):
    """이미지와 텍스트를 처리하여 응답 생성"""
    try:
        # 시스템과 사용자 메시지 설정
        chat = [
            {"role": "system", "content": {"type": "text", "text": "당신은 도움이 되는 비전 언어 모델입니다."}},
            {"role": "user", "content": {"type": "text", "text": user_input}},
        ]

        # 이미지가 제공된 경우, 이미지 추가
        if image is not None:
            chat[1]["content"] = [
                {"type": "text", "text": user_input},
                {"type": "image", "image": image}
            ]

        # 이미지 및 비디오 로드
        new_chat, all_images, is_video_list = preprocessor.load_images_videos(chat)
        preprocessed = preprocessor(all_images, is_video_list=is_video_list)

        # 채팅 템플릿 적용
        input_ids = tokenizer.apply_chat_template(
            new_chat,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True
        )

        # 생성 옵션 설정
        generation_kwargs = {
            "input_ids": input_ids.to(device),
            "max_new_tokens": 1024,
            "do_sample": True,
            "top_p": 0.7,
            "temperature": 0.5,
            "repetition_penalty": 1.0,
            **preprocessed
        }

        # 응답 생성
        output_ids = model.generate(**generation_kwargs)

        # 결과 디코딩
        output = tokenizer.batch_decode(output_ids)[0]

        # 응답 부분만 추출
        if "<|assistant|>" in output:
            return output.split("<|assistant|>")[1].strip()
        else:
            return output.strip()
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"


def chat_interface(message, history, image):
    """채팅 인터페이스 핸들러"""
    response = process_image_and_text(image, message)
    return response


# Gradio 인터페이스 설정
with gr.Blocks(title="CLOVA X 3B 비전 챗봇") as demo:
    gr.Markdown("## CLOVA X 3B 비전 챗봇")
    gr.Markdown("네이버의 HyperCLOVAX-SEED-Vision-Instruct-3B 모델을 이용한 이미지 인식 챗봇입니다.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600)
            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="질문을 입력하세요...",
                    container=False,
                    scale=9
                )
                submit = gr.Button("전송", scale=1)

        with gr.Column(scale=2):
            image_input = gr.Image(type="pil", label="이미지 업로드 (선택사항)")
            clear_btn = gr.Button("대화 초기화")

    # 예제 추가
    gr.Examples(
        examples=[
            ["이 이미지에 무엇이 있나요?", None],
            ["이 이미지를 자세히 설명해주세요.", None],
            ["이 이미지에서 보이는 물체들을 나열해주세요.", None]
        ],
        inputs=[msg, image_input]
    )

    # 이벤트 설정
    submit_event = submit.click(
        chat_interface,
        inputs=[msg, chatbot, image_input],
        outputs=[chatbot]
    ).then(
        lambda: "", None, msg
    )

    msg.submit(
        chat_interface,
        inputs=[msg, chatbot, image_input],
        outputs=[chatbot]
    ).then(
        lambda: "", None, msg
    )

    clear_btn.click(lambda: None, None, chatbot)
    clear_btn.click(lambda: None, None, image_input)

# 실행
if __name__ == "__main__":
    demo.launch(server_port=8283, server_name="0.0.0.0", share=True)