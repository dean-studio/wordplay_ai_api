import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# 모델, 프로세서, 토크나이저 로드
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"기기: {device}")
print(f"모델 로딩 중: {model_name}")

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device=device)
preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("모델 로딩 완료")

# 텍스트 전용 예제
print("\n===== 텍스트 예제 =====")
text_chat = [
    {"role": "system", "content": "너는 도움이 되는 AI 어시스턴트야!"},
    {"role": "user", "content": "안녕하세요, 어떻게 지내세요?"},
    {"role": "assistant", "content": "안녕하세요! 잘 지내고 있어요. 무엇을 도와드릴까요?"},
    {"role": "user", "content": "인공지능에 대해 간단히 설명해줄래요?"},
]

input_ids = tokenizer.apply_chat_template(text_chat, return_tensors="pt", tokenize=True)
input_ids = input_ids.to(device=device)

print("텍스트 생성 중...")
output_ids = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    top_p=0.6,
    temperature=0.5,
    repetition_penalty=1.0,
)

print("=" * 80)
print("텍스트 결과:")
print(tokenizer.batch_decode(output_ids)[0])
print("=" * 80)

# 이미지 예제
print("\n===== 이미지 예제 =====")
image_chat = [
    {"role": "system", "content": {"type": "text", "text": "너는 도움이 되는 비전 언어 AI 어시스턴트야!"}},
    {"role": "user", "content": {"type": "text", "text": "이 이미지에 무엇이 있는지 설명해줘."}},
    {
        "role": "user",
        "content": {
            "type": "image",
            "filename": "example.png",
            "image": "https://github.com/naver-ai/rdnet/blob/main/resources/images/tradeoff_sota.png?raw=true",
        }
    },
]

print("이미지 처리 중...")
new_image_chat, all_images, is_video_list = preprocessor.load_images_videos(image_chat)
preprocessed = preprocessor(all_images, is_video_list=is_video_list)

input_ids = tokenizer.apply_chat_template(
    new_image_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True,
)

output_ids = model.generate(
    input_ids=input_ids.to(device=device),
    max_new_tokens=512,
    do_sample=True,
    top_p=0.7,
    temperature=0.6,
    repetition_penalty=1.0,
    **preprocessed,
)

print("=" * 80)
print("이미지 결과:")
print(tokenizer.batch_decode(output_ids)[0])
print("=" * 80)

# 한국어로 이미지 분석 예제
print("\n===== 한국어 이미지 분석 예제 =====")
korean_image_chat = [
    {"role": "system", "content": {"type": "text", "text": "당신은 도움이 되는 비전 AI 어시스턴트입니다."}},
    {"role": "user", "content": {"type": "text", "text": "이 이미지를 자세히 분석해서 한국어로 설명해주세요."}},
    {
        "role": "user",
        "content": {
            "type": "image",
            "filename": "example2.png",
            "image": "https://github.com/naver-ai/rdnet/blob/main/resources/images/tradeoff.png?raw=true",
        }
    },
]

print("한국어 이미지 분석 처리 중...")
new_korean_chat, korean_images, korean_is_video_list = preprocessor.load_images_videos(korean_image_chat)
korean_preprocessed = preprocessor(korean_images, is_video_list=korean_is_video_list)

korean_input_ids = tokenizer.apply_chat_template(
    new_korean_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True,
)

korean_output_ids = model.generate(
    input_ids=korean_input_ids.to(device=device),
    max_new_tokens=512,
    do_sample=True,
    top_p=0.7,
    temperature=0.6,
    repetition_penalty=1.0,
    **korean_preprocessed,
)

print("=" * 80)
print("한국어 이미지 분석 결과:")
print(tokenizer.batch_decode(korean_output_ids)[0])
print("=" * 80)