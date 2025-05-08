from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
import os
import torch
import gc
import sys

# 메모리 초기화
gc.collect()
torch.cuda.empty_cache()

# TensorBoard 로그 디렉토리 생성
tensorboard_dir = "./tensorboard_logs"
os.makedirs(tensorboard_dir, exist_ok=True)

# KorQuAD V2 데이터셋 로드 및 처리
print("KorQuAD V2 데이터셋 로딩 중...")
try:
    korquad_v2 = load_dataset("KorQuAD/squad_kor_v2", trust_remote_code=True)
    print(f"KorQuAD V2 로드 성공: {len(korquad_v2['train'])} 샘플")
except Exception as e:
    print(f"KorQuAD V2 로드 실패: {e}")
    sys.exit(1)


# 모델 및 토크나이저 로드 - 메모리 관리를 위해 나중에 로드

# 데이터 준비 함수
def convert_to_qa_format(example):
    try:
        # 필수 필드 확인
        if "context" not in example or "question" not in example or "answer" not in example:
            return None

        context = example["context"]
        question = example["question"]

        # 답변 텍스트 추출
        answer_text = ""
        if isinstance(example["answer"], dict) and "text" in example["answer"]:
            answer_text = example["answer"]["text"]

        # 유효한 답변이 있는지 확인
        if not answer_text or not isinstance(answer_text, str):
            return None

        return {
            "context": context,
            "question": question,
            "answer": answer_text
        }
    except Exception as e:
        print(f"샘플 변환 오류: {e}")
        return None


# 일부 데이터만 처리 (메모리 제한)
print("데이터 샘플 선택 중...")
max_samples = 5000  # 5천 샘플로 더 감소
valid_samples = []

for i in range(min(max_samples, len(korquad_v2["train"]))):
    if i % 1000 == 0:
        print(f"샘플 처리 중: {i}/{max_samples}")

    sample = korquad_v2["train"][i]
    processed = convert_to_qa_format(sample)
    if processed:
        valid_samples.append(processed)

# 메모리 정리
del korquad_v2
gc.collect()

# 데이터셋 생성
print(f"유효 샘플 수: {len(valid_samples)}")
dataset = Dataset.from_list(valid_samples)

# 이제 모델 로드
print("토크나이저 로드 중...")
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 특수 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 데이터 포맷팅 함수
def format_qa_for_training(examples):
    formatted_texts = []

    for i in range(len(examples["context"])):
        context = examples["context"][i]
        question = examples["question"][i]
        answer = examples["answer"][i]

        # 지시문 형식으로 변환
        prompt = f"다음 문서를 읽고 질문에 답하세요.\n\n문서: {context}\n\n질문: {question}\n\n답변:"

        # 학습 텍스트 (프롬프트 + 답변)
        full_text = f"{prompt} {answer}"
        formatted_texts.append(full_text)

    return {"text": formatted_texts}


# 데이터 포맷팅
print("데이터 포맷팅 중...")
formatted_dataset = dataset.map(
    format_qa_for_training,
    batched=True,
    batch_size=100,
    remove_columns=dataset.column_names
)


# 토큰화 함수
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )


# 데이터 토큰화
print("데이터 토큰화 중...")
tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=64,
    remove_columns=["text"]
)


# 레이블 설정 (인과적 언어 모델링을 위한 입력 ID 복사)
def set_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples


tokenized_dataset = tokenized_dataset.map(set_labels, batched=True)

# 메모리 정리
del dataset, formatted_dataset
gc.collect()
torch.cuda.empty_cache()

# 모델 로드
print("모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 설정
print("LoRA 설정 중...")
config = LoraConfig(
    r=4,  # 더 작은 rank 값
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # 타겟 모듈 축소
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 모델을 명시적으로 훈련 모드로 설정
model.train()

# 모델 가중치 타입 확인
print(f"모델 가중치 타입: {model.dtype}")

# 모델의 기본 파라미터 고정
for param in model.parameters():
    param.requires_grad = False

# LoRA 어댑터 초기화 전에 훈련 가능한 파라미터 확인
trainable_before = sum(p.requires_grad for p in model.parameters())
print(f"LoRA 적용 전 훈련 가능한 파라미터: {trainable_before}")

# LoRA 어댑터 초기화
model = get_peft_model(model, config)

# LoRA 적용 후 훈련 가능한 파라미터 확인
trainable_after = sum(p.requires_grad for p in model.parameters())
print(f"LoRA 적용 후 훈련 가능한 파라미터: {trainable_after}")

# 데이터 수집기 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./clova-lora-qa",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_dir=tensorboard_dir,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to=["tensorboard"],
    fp16=True,
    push_to_hub=False
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


# 훈련 가능한 파라미터 확인
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"훈련 가능한 파라미터: {trainable_params:,d} / 전체 파라미터: {all_param:,d} ({100 * trainable_params / all_param:.2f}%)")


print_trainable_parameters(model)

# 학습 시작
print("학습 시작...")
trainer.train()

# 모델 저장
model.save_pretrained("./clova-lora-qa-final")
tokenizer.save_pretrained("./clova-lora-qa-final")
print("모델 저장 완료!")