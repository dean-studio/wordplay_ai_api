from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import Trainer
import os
import json
import torch
import gc

# 메모리 초기화
gc.collect()
torch.cuda.empty_cache()

# GPU 메모리 설정
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)  # GPU 메모리 사용량 제한
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# TensorBoard 로그 디렉토리 생성
tensorboard_dir = "./tensorboard_logs"
os.makedirs(tensorboard_dir, exist_ok=True)

# 모델 및 토크나이저 로드 (8비트 양자화 적용)
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,  # 메모리 절약을 위한 8비트 양자화
    device_map="auto"  # 자동 장치 배치
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 특수 토큰 설정 확인
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA 설정 (더 작은 rank 사용)
lora_config = LoraConfig(
    r=8,  # rank 값 감소 (16 -> 8)
    lora_alpha=16,  # alpha 값 감소 (32 -> 16)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# LoRA 모델 생성
peft_model = get_peft_model(model, lora_config)

# KoCommercial-Dataset 로드 (샘플 수 제한)
print("KoCommercial-Dataset 로딩 중...")
ko_commercial = load_dataset("MarkrAI/KoCommercial-Dataset")

# 데이터셋 구조 확인
print("\nKoCommercial 데이터셋 구조 확인:")
sample = ko_commercial["train"][0]
print(f"샘플 키: {list(sample.keys())}")
for key in sample.keys():
    value = sample[key]
    print(f"- {key}: {type(value)}")
    if isinstance(value, list) and value:
        print(f"  - 첫 항목 타입: {type(value[0])}")


# KoCommercial-Dataset 준비 (리스트 처리 추가)
def prepare_ko_commercial(example):
    try:
        # 필드 데이터 타입 처리
        instruction = example.get("instruction", "")
        if isinstance(instruction, list):
            instruction = " ".join(str(item) for item in instruction)
        elif not isinstance(instruction, str):
            instruction = str(instruction)

        input_text = example.get("input", "")
        if isinstance(input_text, list):
            input_text = " ".join(str(item) for item in input_text)
        elif not isinstance(input_text, str):
            input_text = str(input_text)

        output = example.get("output", "")
        if isinstance(output, list):
            output = " ".join(str(item) for item in output)
        elif not isinstance(output, str):
            output = str(output)

        # 프롬프트 생성
        user_content = instruction
        if input_text:
            user_content += f"\n{input_text}"

        # CLOVA 형식으로 변환
        chat = [
            {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]

        # 챗 템플릿 적용
        full_text = tokenizer.apply_chat_template(chat, tokenize=False)
        return {"text": full_text}
    except Exception as e:
        print(f"KoCommercial 샘플 처리 중 오류: {str(e)}")
        # 오류 발생 시 기본 텍스트 반환
        return {"text": "오류 발생"}


print("\nKoCommercial 데이터 변환 중...")
ko_commercial_formatted = []

# 안전한 변환을 위해 배치 처리 대신 개별 처리
for i, example in enumerate(ko_commercial["train"]):
    try:
        processed = prepare_ko_commercial(example)
        if processed["text"] != "오류 발생":
            ko_commercial_formatted.append(processed)
    except Exception as e:
        print(f"샘플 {i} 처리 실패: {str(e)}")

    if i % 10000 == 0 and i > 0:
        print(f"{i}개 샘플 처리 완료...")

# 리스트를 데이터셋으로 변환
ko_commercial_formatted = Dataset.from_list(ko_commercial_formatted)
print(f"변환된 KoCommercial 데이터: {len(ko_commercial_formatted)} 샘플")

# KorQuAD V2 데이터셋 로드 및 처리
print("\nKorQuAD V2 데이터셋 로드 중...")
try:
    # KorQuAD V2 로드
    korquad_v2 = load_dataset("KorQuAD/squad_kor_v2", trust_remote_code=True)

    # 샘플 확인
    print("\nKorQuAD V2 구조 확인:")
    sample = korquad_v2["train"][0]
    print(f"샘플 키: {list(sample.keys())}")


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
    korquad_formatted = []

    # 안전한 변환을 위해 배치 처리 대신 개별 처리
    for i, example in enumerate(korquad_v2["train"]):
        try:
            processed = process_korquad_v2(example)
            if processed is not None:
                korquad_formatted.append(processed)
        except Exception as e:
            pass

        if i % 10000 == 0 and i > 0:
            print(f"{i}개 샘플 처리 완료...")

    # 리스트를 데이터셋으로 변환
    korquad_formatted = Dataset.from_list(korquad_formatted)
    print(f"변환된 KorQuAD 데이터: {len(korquad_formatted)} 샘플")

    # 데이터셋 병합
    print("\nKoCommercial과 KorQuAD 데이터셋 병합 중...")
    combined_dataset = concatenate_datasets([ko_commercial_formatted, korquad_formatted])
    print(f"병합된 데이터셋 크기: {len(combined_dataset)} 샘플")

except Exception as e:
    print(f"\nKorQuAD V2 처리 중 오류 발생: {str(e)}")
    print("KoCommercial-Dataset만 사용합니다.")
    combined_dataset = ko_commercial_formatted

# 메모리 정리
gc.collect()
torch.cuda.empty_cache()


# 토큰화 및 데이터 셋업 (더 작은 max_length 사용)
def tokenize_function(examples):
    results = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,  # 시퀀스 길이 감소 (2048 -> 1024)
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
    batch_size=64,  # 배치 크기 감소로 메모리 사용량 제한
    remove_columns=combined_dataset.column_names
)

# 메모리 정리
gc.collect()
torch.cuda.empty_cache()

# 데이터 수집기 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 인과적 언어 모델링
)

# 학습 설정 (더 작은 배치 크기 사용)
print("\n학습 설정 구성 중...")
training_args = TrainingArguments(
    output_dir="./clova-lora-combined",
    per_device_train_batch_size=1,  # 배치 크기 감소 (4 -> 1)
    gradient_accumulation_steps=16,  # 그라디언트 누적 단계 증가 (4 -> 16)
    learning_rate=2e-4,
    num_train_epochs=1,
    save_steps=2000,

    # TensorBoard 설정
    logging_dir=tensorboard_dir,
    logging_strategy="steps",
    logging_steps=100,
    report_to=["tensorboard"],

    # 메모리 최적화 설정
    fp16=True,
    optim="adamw_torch",
    gradient_checkpointing=True,  # 그라디언트 체크포인팅 활성화
    remove_unused_columns=False,

    # 디스크 공간 관리
    save_total_limit=3,  # 최대 3개 체크포인트만 저장
)

# Trainer 초기화
print("\nTrainer 초기화 중...")
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
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