from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer

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
ko_commercial = load_dataset("MarkrAI/KoCommercial-Dataset")
korquad_v2 = load_dataset("KorQuAD/squad_kor_v2")


# KorQuAD 데이터셋 형식 변환 함수
def convert_korquad_format(example):
    # 지시문 형식으로 변환
    instruction = f"다음 문서를 읽고 질문에 답하세요.\n\n문서: {example['context']}\n\n질문: {example['question']}"

    # 답변이 리스트면 첫 번째 항목 사용
    answer_text = example['answers']['text'][0] if isinstance(example['answers']['text'], list) else example['answers'][
        'text']

    return {
        "instruction": instruction,
        "input": "",
        "output": answer_text
    }


# KorQuAD 데이터셋 형식 변환
korquad_v2_converted = korquad_v2.map(convert_korquad_format)

# 두 데이터셋 합치기 (필요한 필드만 선택)
# 두 데이터셋의 구조가 다를 수 있으므로 필요한 필드만 선택
ko_commercial_subset = ko_commercial["train"].select_columns(["instruction", "input", "output"])
korquad_v2_subset = korquad_v2_converted["train"].select_columns(["instruction", "input", "output"])

# 데이터셋 병합
combined_dataset = concatenate_datasets([ko_commercial_subset, korquad_v2_subset])


# 형식에 맞게 변환
def format_for_clova(example):
    chat = [
        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
        {"role": "user", "content": example["instruction"] + (f"\n{example['input']}" if example["input"] else "")},
        {"role": "assistant", "content": example["output"]}
    ]
    return {"text": tokenizer.apply_chat_template(chat, tokenize=False)}


formatted_dataset = combined_dataset.map(format_for_clova)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./clova-lora-with-korquad",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=15000,  # 데이터셋이 커졌으므로 더 많은 스텝
    save_steps=1000,
    logging_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
)

# 트레이너 설정
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_text_field="text",
)

# 학습 시작
trainer.train()

# 모델 저장
peft_model.save_pretrained("./clova-lora-korquad-final")