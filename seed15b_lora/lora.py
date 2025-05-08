from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import Trainer
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

# 특수 토큰 설정 확인
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# KoCommercial-Dataset 로드
print("KoCommercial-Dataset 로딩 중...")
ko_commercial = load_dataset("MarkrAI/KoCommercial-Dataset")
print(f"KoCommercial-Dataset 로드 성공: {len(ko_commercial['train'])} 샘플")


# KoCommercial-Dataset 준비
def prepare_ko_commercial(example):
    # CLOVA 형식으로 변환
    chat = [
        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
        {"role": "user",
         "content": example.get("instruction", "") + (f"\n{example.get('input', '')}" if example.get("input") else "")},
        {"role": "assistant", "content": example.get("output", "")}
    ]
    # 챗 템플릿 적용
    full_text = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": full_text}


print("\nKoCommercial 데이터 변환 중...")
ko_commercial_formatted = ko_commercial["train"].map(prepare_ko_commercial)
print(f"변환된 KoCommercial 데이터: {len(ko_commercial_formatted)} 샘플")

# KorQuAD V2 데이터셋 로드 및 처리
print("\nKorQuAD V2 데이터셋 로드 중...")
try:
    # KorQuAD V2 로드
    korquad_v2 = load_dataset("KorQuAD/squad_kor_v2", trust_remote_code=True)
    print(f"KorQuAD V2 로드 성공: {len(korquad_v2['train'])} 샘플")

    # 샘플 확인
    print("\nKorQuAD V2 구조 확인:")
    sample = korquad_v2["train"][0]
    print(f"샘플 키: {list(sample.keys())}")
    print(f"답변 구조: {sample['answer'].keys() if isinstance(sample.get('answer'), dict) else 'Not a dict'}")


    # 정확한 구조에 맞게 변환 함수 수정
    def process_korquad_v2(example):
        try:
            # 필수 필드 확인
            if "context" not in example or "question" not in example or "answer" not in example:
                return None

            context = example["context"]
            question = example["question"]

            # 명확한 구조에 따라 answer 필드 처리
            answer_text = ""
            if isinstance(example["answer"], dict) and "text" in example["answer"]:
                answer_text = example["answer"]["text"]

            # 유효한 답변이 있는지 확인
            if not answer_text or not isinstance(answer_text, str):
                return None

            # 지시문 형식으로 변환
            instruction = f"다음 문서를 읽고 질문에 답하세요.\n\n문서: {context}\n\n질문: {question}"

            # 챗 형식으로 변환
            chat = [
                {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": answer_text}
            ]

            # 템플릿 적용
            full_text = tokenizer.apply_chat_template(chat, tokenize=False)
            return {"text": full_text}

        except Exception as e:
            return None


    print("\nKorQuAD V2 데이터 변환 중...")
    korquad_processed = korquad_v2["train"].map(
        process_korquad_v2,
        remove_columns=korquad_v2["train"].column_names
    )

    # None 값 필터링
    korquad_formatted = korquad_processed.filter(lambda example: example["text"] is not None)
    print(f"변환된 KorQuAD 데이터: {len(korquad_formatted)} 샘플")

    # 데이터셋 병합
    print("\nKoCommercial과 KorQuAD 데이터셋 병합 중...")
    combined_dataset = concatenate_datasets([ko_commercial_formatted, korquad_formatted])
    print(f"병합된 데이터셋 크기: {len(combined_dataset)} 샘플")

except Exception as e:
    print(f"\nKorQuAD V2 처리 중 오류 발생: {str(e)}")
    print("KoCommercial-Dataset만 사용합니다.")
    combined_dataset = ko_commercial_formatted


# 토큰화 및 데이터 셋업
def tokenize_function(examples):
    results = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt"
    )
    # 언어 모델링을 위한 레이블 설정
    results["labels"] = results["input_ids"].clone()
    return results


print("\n데이터셋 토큰화 중...")
tokenized_dataset = combined_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=combined_dataset.column_names
)

# 데이터 수집기 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 인과적 언어 모델링
)

# 학습 설정
print("\n학습 설정 구성 중...")
training_args = TrainingArguments(
    output_dir="./clova-lora-combined",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    save_steps=1000,

    # TensorBoard 설정
    logging_dir=tensorboard_dir,
    logging_strategy="steps",
    logging_steps=10,
    report_to=["tensorboard"],

    # 기타 설정
    fp16=torch.cuda.is_available(),
    optim="adamw_torch",
    remove_unused_columns=False,
)

# Trainer 초기화
print("\nTrainer 초기화 중...")
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 학습 시작
print("\n학습 시작 중...")
print(f"데이터셋 크기: {len(tokenized_dataset)}")
print(f"TensorBoard 로그 위치: {tensorboard_dir}")
trainer.train()

# 모델 저장
print("\n모델 저장 중...")
peft_model.save_pretrained("./clova-lora-combined")
tokenizer.save_pretrained("./clova-lora-combined")
print("\n모든 과정이 완료되었습니다!")