import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# 모델, 토크나이저 로드
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"기기: {device}")
print(f"모델 로딩 중: {model_name}")

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("모델 로딩 완료")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    model = None
    tokenizer = None


# 채팅 처리 함수
def chat_with_clova(message, history):
    """Gradio 채팅 인터페이스용 함수"""
    if model is None or tokenizer is None:
        return "모델 로딩에 실패했습니다. 콘솔 로그를 확인해주세요."

    # 대화 기록 포맷 변환
    formatted_history = [
        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."}
    ]

    # 대화 기록을 모델 입력 형식으로 변환
    for user_msg, bot_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        if bot_msg:  # 빈 응답이 아닌 경우에만 추가
            formatted_history.append({"role": "assistant", "content": bot_msg})

    # 현재 메시지 추가
    formatted_history.append({"role": "user", "content": message})

    # 모델 입력 생성
    input_ids = tokenizer.apply_chat_template(formatted_history, return_tensors="pt", tokenize=True)
    input_ids = input_ids.to(device=device)

    # 응답 생성
    output_ids = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.7,
        temperature=0.6,
        repetition_penalty=1.0,
    )

    response = tokenizer.batch_decode(output_ids)[0]

    # 응답에서 어시스턴트 부분 추출
    if "<|assistant|>" in response:
        assistant_response = response.split("<|assistant|>")[-1].strip()
    else:
        # 응답 구조가 다르면 사용자 입력 이후부터 추출 시도
        try:
            assistant_response = response.split(message)[-1].strip()
        except:
            assistant_response = response

    return assistant_response


# Gradio 인터페이스 설정
def create_gradio_interface():
    with gr.Blocks(title="CLOVA X 챗봇") as demo:
        gr.Markdown("# CLOVA X 텍스트 챗봇")
        gr.Markdown(f"네이버의 HyperCLOVAX-SEED-Vision-Instruct-3B 모델을 이용한 텍스트 챗봇입니다.")

        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(
            show_label=False,
            placeholder="메시지를 입력하세요...",
            container=False
        )

        with gr.Row():
            submit_btn = gr.Button("전송")
            clear_btn = gr.Button("대화 초기화")

        # 예제 추가
        gr.Examples(
            examples=[
                "안녕하세요, 자기소개 부탁해요.",
                "한국의 유명한 관광지 추천해주세요.",
                "인공지능에 대해 간단히 설명해줄래요?",
                "서울에서 데이트하기 좋은 장소는 어디인가요?",
                "요즘 인기있는 영화 추천해주세요.",
            ],
            inputs=msg
        )

        # 이벤트 설정
        msg.submit(chat_with_clova, [msg, chatbot], [chatbot]).then(
            lambda: "", None, msg
        )

        submit_btn.click(chat_with_clova, [msg, chatbot], [chatbot]).then(
            lambda: "", None, msg
        )

        clear_btn.click(lambda: [], None, chatbot)

    return demo


# Gradio 인터페이스 실행
if __name__ == "__main__":
    gradio_interface = create_gradio_interface()
    gradio_interface.launch(server_port=8283, server_name="0.0.0.0", share=True)