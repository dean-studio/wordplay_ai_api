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

# ==================================================
# 1. 모델 및 토크나이저 로드/저장 함수
# ==================================================
MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
LOCAL_PATH = "./models/koalpaca-5.8b"


def load_or_download_model(model_name: str, local_path: str):
    if not os.path.exists(local_path):
        logger.info("로컬 모델 경로가 존재하지 않습니다. 모델 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        os.makedirs(local_path, exist_ok=True)
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
    else:
        logger.info("로컬 모델 경로가 존재합니다. 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("토크나이저 로드 완료. PAD 토큰을 EOS 토큰으로 설정함.")
    return tokenizer, local_path


def load_model(tokenizer, local_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    logger.info("4-bit 양자화 설정 완료 (BitsAndBytesConfig).")

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    logger.info("모델 로드 완료 (4-bit 양자화, 자동 device mapping).")

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    logger.info("모델이 k-bit 훈련에 맞게 준비됨 (캐시 비활성화, gradient checkpointing 활성화).")
    return model


# ==================================================
# 2. LoRA 설정 및 적용
# ==================================================
def apply_lora(model):
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
    return model


# ==================================================
# 3. MySQL 데이터 로드 및 전처리
# ==================================================
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


def preprocess_data(df: pd.DataFrame):
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
    logger.info("생성된 튜닝 레코드 수: %d", len(records))
    df_records = pd.DataFrame(records)
    logger.debug("튜닝 레코드 샘플:\n%s", df_records.head().to_string())
    dataset = Dataset.from_pandas(df_records)
    logger.info("HuggingFace Dataset 생성 완료. 전체 예제 수: %d", len(dataset))
    return dataset


def tokenize_fn(example, tokenizer):
    prompt = f"<s>{example['instruction']}\n답변: {example['output']}</s>"
    tokenized_output = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    return tokenized_output


# ==================================================
# 4. Trainer 및 학습
# ==================================================
def train_model(model, tokenizer, tokenized_dataset):
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

    logger.info("훈련 시작...")
    trainer.train()
    logger.info("훈련 완료.")
    return trainer


def save_adapter(model, tokenizer, output_dir="./output/koalpaca-lora"):
    from safetensors.torch import save_file
    os.makedirs(output_dir, exist_ok=True)
    logger.info("모델 저장 디렉토리 생성 완료: %s", output_dir)

    logger.info("튜닝된 모델 상태 (safetensors) 저장 중...")
    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
    logger.info("safetensors 저장 완료.")

    logger.info("PEFT 설정 및 어댑터 저장 중...")
    model.peft_config[model.active_adapter].save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("PEFT 어댑터와 토크나이저 저장 완료.")


# ==================================================
# 5. 메인 흐름: 로드 → LoRA 적용 → 데이터 전처리 → 토크나이즈 → 학습 → 저장
# ==================================================
def main():
    # 모델 다운로드 또는 로드
    tokenizer, local_path = load_or_download_model(MODEL_NAME, LOCAL_PATH)
    model = load_model(tokenizer, local_path)

    # (원하는 경우, 타겟 모듈 이름 확인 로그는 유지)
    logger.debug("모델 파라미터 이름 중 타겟 모듈 관련 항목 출력:")
    # for name, _ in model.named_parameters():
    #     if any(substr in name for substr in ["query_key_value", "dense_h_to_4h", "dense_4h_to_h", "dense"]):
    #         logger.debug(name)

    # LoRA 어댑터 적용
    model = apply_lora(model)

    # MySQL 데이터 로드 및 전처리
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
      ) LIMIT 100
    """
    df = load_data_from_mysql(
        sql_query,
        host='wordplayapi.mycafe24.com',
        user='wordplayapi',
        password='Hazbola2021!',
        db='wordplayapi'
    )
    # logger.debug("MySQL DataFrame 샘플:\n%s", df.head().to_string())
    dataset = preprocess_data(df)

    # 토크나이즈
    tokenized_dataset = dataset.map(lambda x: tokenize_fn(x, tokenizer))
    # logger.info("토크나이즈 완료. 예시 토크나이즈 결과: %s", tokenized_dataset[0])

    # 학습
    trainer = train_model(model, tokenizer, tokenized_dataset)

    # 저장
    save_adapter(model, tokenizer)


if __name__ == "__main__":
    main()
