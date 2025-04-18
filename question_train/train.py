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
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------
# 모델 및 토크나이저 설정
# ----------------------------
MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
LOCAL_PATH = "./models/koalpaca-5.8b"

if not os.path.exists(LOCAL_PATH):
    logger.info("로컬 모델 경로가 존재하지 않습니다. 모델 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    os.makedirs(LOCAL_PATH, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_PATH)
    model.save_pretrained(LOCAL_PATH)
else:
    logger.info("로컬 모델 경로가 존재합니다. 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
logger.info("토크나이저 로드 완료. PAD 토큰을 EOS 토큰으로 설정함.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
logger.info("4-bit 양자화 설정 완료 (BitsAndBytesConfig).")

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
logger.info("모델 로드 완료 (4-bit 양자화, 자동 device mapping).")

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.gradient_checkpointing_enable()
logger.info("모델이 k-bit 훈련에 맞게 준비됨 (캐시 비활성화, gradient checkpointing 활성화).")

# 디버깅: target 모듈 관련 파라미터 확인 (대상으로 지정한 문자열이 포함된 항목 출력)
logger.debug("모델 파라미터 이름 중 타겟 모듈 관련 항목 출력:")
for name, _ in model.named_parameters():
    if any(substr in name for substr in ["query_key_value", "dense_h_to_4h", "dense_4h_to_h", "dense"]):
        logger.debug(name)

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
logger.info("LoRA 설정 생성: target_modules=%s", peft_config.target_modules)

model = get_peft_model(model, peft_config)
logger.info("LoRA 어댑터 적용 완료. 현재 활성 어댑터: %s", model.active_adapter)

# ----------------------------
# MySQL에서 데이터 로드
# ----------------------------
def load_data_from_mysql(query, host, user, password, db, charset="utf8mb4"):
    logger.info("MySQL 데이터베이스 (%s)에 연결 중...", host)
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        db=db,
        charset=charset
    )
    try:
        df = pd.read_sql(query, conn)
        logger.info("MySQL에서 데이터 로드 완료. 행 수: %d", len(df))
    except Exception as e:
        logger.exception("MySQL 데이터 로드 실패: %s", e)
        raise
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

logger.debug("MySQL DataFrame 샘플:\n%s", df.head().to_string())

# ----------------------------
# 데이터 전처리 및 Dataset 생성
# ----------------------------
records = []
for idx, row in df.iterrows():
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

logger.info("생성된 튜닝 레코드 수: %d", len(records))
df_records = pd.DataFrame(records)
logger.debug("튜닝 레코드 샘플:\n%s", df_records.head().to_string())

dataset = Dataset.from_pandas(df_records)
logger.info("HuggingFace Dataset 생성 완료. 전체 예제 수: %d", len(dataset))

def tokenize(example):
    prompt = f"<s>{example['instruction']}\n답변: {example['output']}</s>"
    tokenized_output = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    return tokenized_output

tokenized_dataset = dataset.map(tokenize)
logger.info("토크나이즈 완료. 예시 토크나이즈 결과: %s", tokenized_dataset[0])

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
logger.info("훈련 인자 설정 완료: %s", training_args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
logger.info("Trainer 초기화 완료.")

# ----------------------------
# 훈련
# ----------------------------
logger.info("훈련 시작...")
trainer.train()
logger.info("훈련 완료.")

# ----------------------------
# 튜닝된 모델(LoRA 어댑터) 저장
# ----------------------------
from safetensors.torch import save_file

OUTPUT_DIR = "./output/koalpaca-lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info("모델 저장 디렉토리 생성 완료: %s", OUTPUT_DIR)

logger.info("튜닝된 모델 상태 (safetensors) 저장 중...")
state_dict = model.state_dict()
save_file(state_dict, os.path.join(OUTPUT_DIR, "adapter_model.safetensors"))
logger.info("safetensors 저장 완료.")

logger.info("PEFT 설정 및 어댑터 저장 중...")
model.peft_config[model.active_adapter].save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logger.info("PEFT 어댑터와 토크나이저 저장 완료.")
