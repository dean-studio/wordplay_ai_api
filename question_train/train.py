import os
import pymysql
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
import torch

# ----------------------------
# 모델 및 토크나이저 설정
# ----------------------------
MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
LOCAL_PATH = "./models/koalpaca-5.8b"

# 로컬에 모델이 저장되어 있지 않다면 다운로드 후 저장
if not os.path.exists(LOCAL_PATH):
    print("로컬 모델 경로가 존재하지 않습니다. 모델 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    os.makedirs(LOCAL_PATH, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_PATH)
    model.save_pretrained(LOCAL_PATH)
else:
    print("로컬 모델 경로가 존재합니다. 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정

# 4-bit 양자화 설정 (BitsAndBytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로드 (양자화 & 자동 device mapping)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

# k-bit 훈련에 맞게 모델 준비
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ----------------------------
# LoRA 설정 및 적용
# ----------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# ----------------------------
# MySQL에서 데이터 로드
# ----------------------------
def load_data_from_mysql(query, host, user, password, db, charset="utf8mb4"):
    """MySQL에서 쿼리를 실행하여 데이터를 DataFrame으로 반환"""
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        db=db,
        charset=charset
    )
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df

sql_query = """
SELECT 
  category,
  grade_level,
  difficulty,
  multiple_choice_question AS mc_question,
  multiple_choice_explanation AS mc_explanation,
  ox_quiz_question AS ox_question,
  ox_quiz_explanation AS ox_explanation
FROM questions
WHERE 
  (
    (multiple_choice_question IS NOT NULL AND multiple_choice_explanation IS NOT NULL)
    OR
    (ox_quiz_question IS NOT NULL AND ox_quiz_explanation IS NOT NULL)
  ) LIMIT 10
"""

df = load_data_from_mysql(
    sql_query,
    host='wordplayapi.mycafe24.com',
    user='wordplayapi',
    password='Hazbola2021!',
    db='wordplayapi'
)

print(df)

# ----------------------------
# 데이터 전처리 및 Dataset 생성
# ----------------------------
records = []
for _, row in df.iterrows():
    meta = f"카테고리: {row['category']} / 학년: {row['grade_level']} / 난이도: {row['difficulty']}"
    if pd.notnull(row['mc_question']) and pd.notnull(row['mc_explanation']):
        records.append({
            "instruction": f"질문: {row['mc_question']}\n{meta}",
            "output": row['mc_explanation']
        })
    if pd.notnull(row['ox_question']) and pd.notnull(row['ox_explanation']):
        records.append({
            "instruction": f"질문: {row['ox_question']}\n{meta}",
            "output": row['ox_explanation']
        })

df_records = pd.DataFrame(records)
dataset = Dataset.from_pandas(df_records)

def tokenize(example):
    # 토크나이저에 의해 최대 길이 512로 트렁케이션 및 패딩 처리
    prompt = f"<s>{example['instruction']}\n답변: {example['output']}</s>"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize)

# ----------------------------
# 훈련 인자 및 Trainer 설정
# ----------------------------
training_args = TrainingArguments(
    output_dir="./output",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# ----------------------------
# 훈련된 모델(LoRA 어댑터) 저장
# ----------------------------
from safetensors.torch import save_file

OUTPUT_DIR = "./output/koalpaca-lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모델 상태 저장 (safetensors 형식)
state_dict = model.state_dict()
save_file(state_dict, os.path.join(OUTPUT_DIR, "adapter_model.safetensors"))
# PEFT 설정 및 어댑터 저장
model.peft_config[model.active_adapter].save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
