from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import Trainer
import os
import torch
import gc

# 메모리 초기화
gc.collect()
torch.cuda.empty_cache()

# TensorBoard 로그 디렉토리 생성
tensorboard_dir = "./tensorboard_logs"
os.makedirs(tensorboard_dir, exist_ok=True)

# 모델 및 토크나이저 로드
print("토크나이저 로딩 중...")
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 특수 토큰 설정 확인
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# KorQuAD V2 데이터셋 로드 및 처리
print("KorQuAD V2 데이터셋 로딩 중...")
korquad_v2 = load_dataset("KorQuAD/squad_kor_v2", trust_remote_code=True)
print(f"KorQuAD V2 로드 성공: {len(korquad_v2['train'])} 샘플")


# KorQuAD 데이터 준비
def prepare_korquad_sample(example):
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


# 일부 데이터만 사용 (메모리 문제 방지)
print("KorQuAD 데이터 선택 및 처리 중...")
sample_count = 10000  # 1만 샘플로 더 감소
korquad_samples = []

for i in range(sample_count):
    if i % 1000 == 0:
        print(f"KorQuAD 처리 중: {i}/{sample_count}")

    sample = korquad_v2["train"][i]
    processed = prepare_korquad_sample(sample)
    if processed is not None:
        korquad_samples.append(processed)

# 데이터셋 생성
print(f"처리된 KorQuAD 샘플 수: {len(korquad_samples)}")
korquad_dataset = Dataset.from_list(korquad_samples)

# 메모리 정리
del korquad_v2
del korquad_samples
gc.collect()
torch.cuda.empty_cache()


# 토큰화 함수
def tokenize_function(examples):
    results = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # 짧은 시퀀스 길이
        padding="max_length",
        return_tensors="pt"
    )
    results["labels"] = results["input_ids"].clone()
    return results


print("데이터셋 토큰화 중...")
tokenized_dataset = korquad_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=64,
    remove_columns=korquad_dataset.column_names
)

# 메모리 정리
del korquad_dataset
gc.collect()
torch.cuda.empty_cache()

# 이제 모델 로드 (메모리 확보 후) - 4비트 양자화 사용
print("모델 로드 중...")

# 모델 로드 (4비트 양자화는 피하고 기본 fp16 사용)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 설정
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# LoRA 모델 생성
peft_model = get_peft_model(model, lora_config)
print("LoRA 모델 생성 완료")

# 학습 가능한 파라미터만 표시
trainable_params = 0
all_param = 0
for _, param in peft_model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(f"훈련 가능한 파라미터: {trainable_params}")
print(f"모든 파라미터: {all_param}")
print(f"훈련 가능한 비율: {100 * trainable_params / all_param:.2f}%")

# 모듈이 올바르게 초기화되었는지 확인
for name, module in peft_model.named_modules():
    if "lora" in name.lower():
        print(f"LoRA 모듈 확인: {name}")

# 학습 설정
print("학습 설정 구성 중...")
training_args = TrainingArguments(
    output_dir="./clova-lora-korquad-only",
    per_device_train_batch_size=1,  # 배치 크기 감소
    gradient_accumulation_steps=8,  # 그라디언트 누적 증가
    learning_rate=2e-4,
    max_steps=2000,  # 스텝 기반 학습으로 변경
    save_steps=500,
    logging_dir=tensorboard_dir,
    logging_strategy="steps",
    logging_steps=50,
    report_to=["tensorboard"],
    fp16=True,
    optim="adamw_torch",  # 기본 옵티마이저 사용
    gradient_checkpointing=True,
    save_total_limit=3,
    remove_unused_columns=False,
)

# Trainer 초기화
print("Trainer 초기화 중...")
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 학습 시작
print("학습 시작 중...")
print(f"데이터셋 크기: {len(tokenized_dataset)}")
print(f"TensorBoard 로그 위치: {tensorboard_dir}")
trainer.train()

# 모델 저장
print("모델 저장 중...")
peft_model.save_pretrained("./clova-lora-korquad-only")
tokenizer.save_pretrained("./clova-lora-korquad-only")
print("모든 과정이 완료되었습니다!")