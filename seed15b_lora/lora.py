from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Features, Value
from trl import SFTTrainer
import os
import json
import torch

# TensorBoard 로그 디렉토리 생성
tensorboard_dir = "./tensorboard_logs"
os.makedirs(tensorboard_dir, exist_ok=True)

# 모델 및 토크나이저 로드
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# LoRA 모델 생성
peft_model = get_peft_model(model, lora_config)

# 데이터셋 로드
print("KoCommercial-Dataset 로딩 중...")
ko_commercial = load_dataset("MarkrAI/KoCommercial-Dataset")

# 필드 구조 표준화
features = Features({
    'instruction': Value('string'),
    'input': Value('string'),
    'output': Value('string')
})

# KoCommercial 데이터셋의 구조 확인 및 표준화
print("KoCommercial 데이터셋 필드 구조 통일 중...")
try:
    # 필드가 아직 없는 경우 처리
    ko_commercial_standardized = ko_commercial["train"].cast(features)
except Exception as e:
    print(f"구조 변환 중 오류: {str(e)}")
    # 데이터셋 구조 확인
    print("현재 KoCommercial 데이터셋 구조:")
    ko_sample = ko_commercial["train"][0]
    print(f"키: {list(ko_sample.keys())}")


    # 필드 구조에 맞게 변환하는 함수 정의
    def standardize_ko_commercial(example):
        output = {}
        output["instruction"] = example.get("instruction", "")
        output["input"] = example.get("input", "")
        output["output"] = example.get("output", "")
        return output


    # 변환 적용
    ko_commercial_standardized = ko_commercial["train"].map(standardize_ko_commercial).cast(features)

# 필요한 경우 KorQuAD 추가 시도 (옵션)
try_korquad = False  # KorQuAD 사용 여부 설정

if try_korquad:
    try:
        print("\nKorQuAD/squad_kor_v2 로딩 시도...")
        korquad_v2 = load_dataset("KorQuAD/squad_kor_v2", trust_remote_code=True)

        # KorQuAD 구조 확인
        print("\nKorQuAD 샘플 형식 확인:")
        kq_sample = korquad_v2["train"][0]
        print(f"키: {list(kq_sample.keys())}")


        # KorQuAD를 표준 형식으로 변환
        def convert_korquad(example):
            instruction = f"다음 문서를 읽고 질문에 답하세요.\n\n문서: {example.get('context', '')}\n\n질문: {example.get('question', '')}"

            # 답변 추출 시도
            answer_text = ""
            if "answers" in example and isinstance(example["answers"], dict) and "text" in example["answers"]:
                texts = example["answers"]["text"]
                answer_text = texts[0] if isinstance(texts, list) and texts else ""

            return {
                "instruction": instruction,
                "input": "",
                "output": answer_text if answer_text else "답변을 찾을 수 없습니다."
            }


        # 변환 및 병합
        korquad_converted = korquad_v2["train"].map(convert_korquad).cast(features)
        combined_dataset = concatenate_datasets([ko_commercial_standardized, korquad_converted])
        print(f"병합된 데이터셋 크기: {len(combined_dataset)} 샘플")

    except Exception as e:
        print(f"\nKorQuAD 처리 중 오류 발생: {str(e)}")
        print("KoCommercial-Dataset만 사용합니다.")
        combined_dataset = ko_commercial_standardized
else:
    print("\nKorQuAD 사용하지 않음. KoCommercial-Dataset만 사용합니다.")
    combined_dataset = ko_commercial_standardized


# 형식에 맞게 변환
def format_for_clova(example):
    chat = [
        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
        {"role": "user",
         "content": example["instruction"] + (f"\n{example['input']}" if example.get("input", "") else "")},
        {"role": "assistant", "content": example.get("output", "")}
    ]
    return {"text": tokenizer.apply_chat_template(chat, tokenize=False)}


print("\n데이터를 CLOVA 형식으로 변환 중...")
formatted_dataset = combined_dataset.map(format_for_clova)


# 데이터셋을 토큰화하여 학습 준비
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)


print("\n데이터셋 토큰화 중...")
tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

# 학습 설정
print("\n학습 설정 구성 중...")
training_args = TrainingArguments(
    output_dir="./clova-lora-with-korquad",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=15000,
    save_steps=1000,

    # TensorBoard 설정
    logging_dir=tensorboard_dir,
    logging_strategy="steps",
    logging_steps=10,
    report_to=["tensorboard"],

    # 기존 설정
    fp16=torch.cuda.is_available(),
    optim="adamw_torch",  # bitsandbytes 문제가 있으면 기본 옵티마이저 사용
)

# 올바른 SFTTrainer 구성
print("\n트레이너 초기화 중...")
try:
    # 최신 버전의 trl 사용 시 (SFTTrainer 인터페이스가 변경됨)
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
    )
except TypeError as e:
    print(f"SFTTrainer 초기화 오류: {str(e)}")
    print("대안으로 기본 Trainer 사용...")
    # 기본 Trainer 사용 (안전한 대안)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

# 학습 시작
print("\n학습 시작 중...")
print(f"데이터셋 크기: {len(formatted_dataset)}")
print(f"TensorBoard 로그 위치: {tensorboard_dir}")
trainer.train()

# 모델 저장
print("\n모델 저장 중...")
peft_model.save_pretrained("./clova-lora-korquad-final")
print("\n모든 과정이 완료되었습니다!")