import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr
import os

# 원본 모델 및 토크나이저 로드
base_model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# LoRA 모델 경로 설정 (여러 LoRA 어댑터 경로를 리스트로 정의)
lora_paths = [
    "../seed15b_lora/clova-lora-qa-final",  # 기존 QA LoRA 어댑터
    "../seed15b_lora_wp/wpdb-lora-tuned-final"  # wp DB LoRA 어댑터
]

# 모델 로드
print("기본 모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 여러 LoRA 어댑터 순차적으로 적용
for lora_path in lora_paths:
    if os.path.exists(lora_path):
        print(f"LoRA 어댑터 로드 중: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"LoRA 어댑터 {os.path.basename(lora_path)}가 성공적으로 적용되었습니다.")
    else:
        print(f"경고: LoRA 어댑터 경로({lora_path})를 찾을 수 없습니다.")

# 모델을 평가 모드로 설정
model.eval()

# GPU 사용 가능 시 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")


def chat_interface(user_input: str) -> str:
    # 개선된 시스템 메시지 - 더 명확하고 상세한 지시 포함
    system_message = """
- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.
- 학습은 LoRA 기법으로 QA 특화 학습 및 워드플레이 문제 해설 특화 학습이 수행되었다.
- 오늘은 2025년 05월 08일(목)이다.
- 당신은 교육 콘텐츠를 분석하여 JSON 형식의 문제를 정확하게 생성하는 전문가이다.
- 입력된 책 내용을 기반으로 객관식 문제와 OX 문제를 생성하고 아래 정확한 JSON 형식으로 출력해야 한다.
- 출력 형식:
{
  "question": "문제 제목", // 짧고 명확하게 작성
  "category": "카테고리", // 교과 영역(국어, 사회, 과학 등)
  "category_idx": 숫자, // 카테고리 인덱스 (임의의 숫자 할당)
  "grade_level": "등급", // 대상 학년 (초등 1학년~고등 3학년)
  "difficulty": 숫자, // 1(쉬움), 2(보통), 3(어려움)
  "multiple_choice": {
    "question": "객관식 문제", // 명확하고 구체적인 질문
    "options": [
      {"id": 1, "value": "보기1"}, // 항상 4개의 보기 제공
      {"id": 2, "value": "보기2"},
      {"id": 3, "value": "보기3"},
      {"id": 4, "value": "보기4"}
    ],
    "answer": 번호, // 정답 번호(1~4)
    "explanation": "정답 설명" // 왜 이 답이 맞는지 명확히 설명
  },
  "ox_quiz": {
    "question": "OX 퀴즈 질문", // 명확한 참/거짓 질문
    "answer": true/false, // true 또는 false로만 작성
    "explanation": "정답 설명" // 왜 참인지 또는 거짓인지 설명
  },
  "reference_links": { // 참고 자료 링크(필수는 아님)
    "priority_1": "나무위키 링크 (1순위)",
    "priority_2": "네이버 블로그 링크 (2순위)",
    "priority_3": "위키피디아 링크 (3순위)"
  }
}

- 중요: 반드시 모든 필드를 채워야 하며, 정확한 JSON 형식을 지켜야 한다.
- JSON은 하나의 완전한 객체로 출력되어야 하며, 주석은 제거해야 한다.
- 학생들이 이해하기 쉽도록 명확하고 정확한 문제와 설명을 작성해야 한다.
- 입력된 콘텐츠 내용을 기반으로만 문제를 생성해야 한다.
- 문제는 교육적으로 가치 있어야 하며, 핵심 개념이나 지식을 테스트해야 한다.
"""

    # 클로바 X 샘플 템플릿: 개선된 시스템 메시지와 사용자 메시지 구성
    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_message},
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

    # 모델을 통한 응답 생성 - 매개변수 최적화
    with torch.no_grad():  # 추론 시 그라디언트 계산 방지
        output_ids = model.generate(
            **inputs,
            max_length=4096,         # 충분한 길이 허용 (JSON 출력을 위해)
            do_sample=True,          # 샘플링 활성화
            temperature=0.7,         # 낮은 온도로 결정적인 출력
            top_p=0.9,               # 확률 분포의 상위 90%만 고려
            repetition_penalty=1.2,  # 반복 방지를 위한 페널티 증가
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
    gr.Markdown("## 교육 콘텐츠 문제 생성기 - CLOVA X (1.5B)")
    chatbot = gr.Chatbot(label="문제 생성기")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(
            placeholder="교과서나 학습 자료의 내용을 붙여넣으면 자동으로 문제가 생성됩니다.",
            show_label=False,
            lines=8  # 더 큰 텍스트 입력 영역
        )

    # 개선된 예제 입력 추가
    gr.Examples([
        ["# 교과서 내용\n\n희소성: '희소'는 매우 적다는 뜻이에요. '희소성'은 사람들이 가지고 싶은 물건이 부족한 상태를 말해요. 사람들이 가지고 싶은 물건이 부족할수록 그 물건의 가치가 더 높아져요. 희소성이 높으면 물건값이 비싸고요, 희소성이 낮으면 물건값이 싸요.\n\n포켓몬빵이 들어오는 시간에 맞춰 동네 편의점 앞에 줄을 서서 기다리는 사람들을 흔히 볼 수 있어요. 특별한 띠부씰의 경우, 빵보다 훨씬 비싼 가격으로 중고 시장에서 팔리기도 하죠."],
        ["# 과학 수업 내용\n\n지구는 태양계에서 세 번째 행성입니다. 지구의 표면적은 약 5억 1000만 제곱킬로미터이며, 71%가 물로 덮여 있습니다. 지구의 대기는 주로 질소(78%)와 산소(21%)로 구성되어 있으며, 이것이 생명체가 살 수 있는 환경을 만들어줍니다. 지구는 자전축이 23.5도 기울어져 있어서 계절의 변화가 생깁니다."],
        ["# 국어 교과서\n\n관용어란 두 개 이상의 단어가 결합하여 특별한 의미를 나타내는 말입니다. 예를 들어 '발이 넓다'는 관용어는 아는 사람이 많다는 뜻으로 사용됩니다. '손이 크다'는 씀씀이가 후하다는 뜻이며, '눈이 높다'는 안목이 좋다는 뜻입니다. 관용어는 문자 그대로의 의미가 아닌 비유적인 의미로 사용되어 언어 표현을 풍부하게 합니다."],
        ["# 수학 학습지\n\n삼각형의 내각의 합은 항상 180도입니다. 삼각형의 세 내각을 모두 더하면 항상 180도가 됩니다. 예를 들어, 세 각이 각각 30도, 60도, 90도인 삼각형이 있다면, 30 + 60 + 90 = 180이 됩니다. 또한 사각형의 내각의 합은 항상 360도입니다. n각형의 내각의 합은 (n-2) × 180도로 구할 수 있습니다."]
    ], txt)

    txt.submit(respond, inputs=[txt, state], outputs=[txt, chatbot])

# 서버 시작
print("Gradio 서버 시작 중...")
demo.launch(server_port=7860, server_name="0.0.0.0")