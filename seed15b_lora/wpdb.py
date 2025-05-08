from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
import os
import torch
import gc
import sys
import pymysql
import pandas as pd
import json

# 메모리 초기화
gc.collect()
torch.cuda.empty_cache()

# TensorBoard 로그 디렉토리 생성
tensorboard_dir = "./tensorboard_logs"
os.makedirs(tensorboard_dir, exist_ok=True)

# 워드플레이 DB에서 데이터 로드
print("워드플레이 DB에서 데이터 로딩 중...")
try:
    conn = pymysql.connect(
        host="wordplayapi.mycafe24.com",
        user="wordplayapi",
        password="Hazbola2021!",
        db="wordplayapi",
        charset="utf8mb4"
    )

    # 워드플레이 문제 데이터 쿼리
    sql_query = """
    SELECT 
        question, category, grade_level, difficulty, 
        multiple_choice_question, multiple_choice_options, 
        multiple_choice_answer, multiple_choice_explanation, 
        ox_quiz_question, ox_quiz_answer, ox_quiz_explanation
    FROM questions 
    LIMIT 3000
    """

    # 데이터 로드
    wp_data = pd.read_sql(sql_query, conn)
    print(f"워드플레이 DB에서 {len(wp_data)} 개의 샘플 로드 완료")
    conn.close()
except Exception as e:
    print(f"워드플레이 DB 로드 실패: {e}")
    sys.exit(1)


# 데이터 준비 함수 - 객관식/OX 문제 처리
def convert_to_samples():
    samples = []

    for idx, row in wp_data.iterrows():
        if idx % 500 == 0:
            print(f"샘플 처리 중: {idx}/{len(wp_data)}")

        # 기본 헤더 정보
        base_info = (
            f"제목: {row['question']}\n"
            f"카테고리: {row['category']}\n"
            f"학년: {row['grade_level']}\n"
            f"난이도: {row['difficulty']}\n"
        )
        prompt_instr = "아래 문제를 읽고 올바른 설명을 작성하세요.\n"

        # 객관식 문제 샘플 생성
        if (pd.notnull(row['multiple_choice_question']) and
                pd.notnull(row['multiple_choice_options']) and
                pd.notnull(row['multiple_choice_answer']) and
                pd.notnull(row['multiple_choice_explanation'])):

            try:
                options = json.loads(row['multiple_choice_options'])
                options_text = "\n".join([f"   {opt['id']}. {opt['value']}" for opt in options])
                answer_option = next(
                    (opt['value'] for opt in options if str(opt['id']) == str(row['multiple_choice_answer'])),
                    "정답 정보 없음")
            except Exception as e:
                print(f"객관식 보기 파싱 실패: {e}")
                options_text = row['multiple_choice_options']
                answer_option = row['multiple_choice_answer']

            sample_text = (
                f"User: {prompt_instr}{base_info}"
                f"문제: {row['multiple_choice_question']}\n"
                f"보기:\n{options_text}\n"
                f"정답: {answer_option}\n\n"
                f"Assistant: {row['multiple_choice_explanation']}"
            )

            samples.append({"text": sample_text})

        # OX 문제 샘플 생성
        if (pd.notnull(row['ox_quiz_question']) and
                pd.notnull(row['ox_quiz_answer']) and
                pd.notnull(row['ox_quiz_explanation'])):
            sample_text = (
                f"User: {prompt_instr}{base_info}"
                f"문제: {row['ox_quiz_question']}\n"
                f"정답: {row['ox_quiz_answer']}\n\n"
                f"Assistant: {row['ox_quiz_explanation']}"
            )
            samples.append({"text": sample_text})

    return samples


# 샘플 변환
print("데이터 샘플 생성 중...")
samples = convert_to_samples()

# 메모리 정리
del wp_data
gc.collect()

# 데이터셋 생성
print(f"유효 샘플 수: {len(samples)}")
dataset = Dataset.from_list(samples)

# 이제 모델 로드
print("토크나이저 로드 중...")
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 특수 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


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
tokenized_dataset = dataset.map(
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

# 훈련/평가 데이터셋 분할 (90%:10%)
splits = tokenized_dataset.train_test_split(train_size=0.9, seed=42)
train_dataset = splits['train']
eval_dataset = splits['test']
print(f"학습 데이터셋: {len(train_dataset)}개, 평가 데이터셋: {len(eval_dataset)}개")

# 메모리 정리
del dataset, tokenized_dataset, samples
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
    r=8,  # 랭크 값
    lora_alpha=32,  # 스케일링 파라미터
    target_modules=["q_proj", "v_proj"],  # 타겟 모듈
    lora_dropout=0.1,
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
    output_dir="./wpdb-lora-tuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    num_train_epochs=15,
    logging_dir=tensorboard_dir,
    logging_steps=25,
    save_steps=100,
    eval_steps=50,
    # 여기를 수정
    # evaluation_strategy="steps",  # 이렇게 되어 있으면 오류 발생
    eval_strategy="steps",  # 올바른 매개변수 이름
    save_total_limit=3,
    remove_unused_columns=False,
    report_to=["tensorboard"],
    fp16=True,
    push_to_hub=False,
    do_eval=True,
    load_best_model_at_end=True,
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
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
model.save_pretrained("./wpdb-lora-tuned-final")
tokenizer.save_pretrained("./wpdb-lora-tuned-final")
print("모델 저장 완료!")