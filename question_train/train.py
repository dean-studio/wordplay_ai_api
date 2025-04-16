import pymysql
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
import torch
import os


model_name = "beomi/KoAlpaca-Polyglot-5.8B"
local_path = "./models/koalpaca-5.8b"

if not os.path.exists(local_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)

tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    local_path,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)


model = get_peft_model(model, peft_config)

### ✅ MySQL → pandas
conn = pymysql.connect(
    host='wordplayapi.mycafe24.com',
    user='wordplayapi',
    password='Hazbola2021!',
    db='wordplayapi',
    charset='utf8mb4'
)

query = """
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

df = pd.read_sql(query, conn)

print(df)

conn.close()

### ✅ 객체 리스트 만들기
records = []

for _, row in df.iterrows():
    meta = f"카테고리: {row.category} / 학년: {row.grade_level} / 난이도: {row.difficulty}"

    if pd.notnull(row.mc_question) and pd.notnull(row.mc_explanation):
        records.append({
            "instruction": f"질문: {row.mc_question}\n{meta}",
            "output": row.mc_explanation
        })

    if pd.notnull(row.ox_question) and pd.notnull(row.ox_explanation):
        records.append({
            "instruction": f"질문: {row.ox_question}\n{meta}",
            "output": row.ox_explanation
        })

### ✅ HuggingFace Dataset으로 변환
dataset = Dataset.from_pandas(pd.DataFrame(records))

def tokenize(example):
    prompt = f"<s>{example['instruction']}\n답변: {example['output']}</s>"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("./output/koalpaca-lora")
tokenizer.save_pretrained("./output/koalpaca-lora")