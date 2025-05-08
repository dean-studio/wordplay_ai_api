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


    # 개선된 데이터 추출 함수
    def extract_korquad_v2(examples):
        records = []
        count = 0
        errors = 0

        for idx, example in enumerate(examples):
            try:
                # 기본 필드 확인
                if "context" not in example or "question" not in example:
                    continue

                # 컨텍스트 및 질문 추출
                context = example["context"]
                question = example["question"]

                # 답변 텍스트 추출 시도 - 개선된 로직
                answer_text = ""
                if "answers" in example:
                    answers = example["answers"]
                    if isinstance(answers, dict) and "text" in answers:
                        text_value = answers["text"]
                        if isinstance(text_value, list) and text_value:
                            answer_text = str(text_value[0])
                        elif isinstance(text_value, str):
                            answer_text = text_value
                        elif isinstance(text_value, dict):
                            # 딕셔너리인 경우 첫 번째 값 사용
                            if text_value:
                                first_key = next(iter(text_value))
                                answer_text = str(text_value[first_key])
                        else:
                            answer_text = str(text_value)
                    elif isinstance(answers, list) and answers:
                        if isinstance(answers[0], dict) and "text" in answers[0]:
                            text_item = answers[0]["text"]
                            if isinstance(text_item, str):
                                answer_text = text_item
                            else:
                                answer_text = str(text_item)

                if not answer_text and "answer" in example:
                    answer_value = example["answer"]
                    if isinstance(answer_value, str):
                        answer_text = answer_value
                    else:
                        answer_text = str(answer_value)

                # 유효한 문자열 답변이 있는 경우에만 추가
                if answer_text and isinstance(answer_text, str):
                    answer_text = answer_text.strip()
                    if len(answer_text) > 0:
                        # 지시문 형식 변환
                        instruction = f"다음 문서를 읽고 질문에 답하세요.\n\n문서: {context}\n\n질문: {question}"

                        # 챗 형식으로 변환
                        chat = [
                            {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": answer_text}
                        ]

                        # 템플릿 적용
                        full_text = tokenizer.apply_chat_template(chat, tokenize=False)
                        records.append({"text": full_text})
                        count += 1

                        if count % 5000 == 0:
                            print(f"KorQuAD 데이터 {count}개 처리 완료...")

            except Exception as e:
                errors += 1
                if errors % 1000 == 0:
                    print(f"총 {errors}개 샘플 처리 중 오류 발생")
                continue

        return records


    print("\nKorQuAD V2 데이터 추출 중...")
    qa_records = extract_korquad_v2(korquad_v2["train"])
    print(f"추출된 KorQuAD 데이터: {len(qa_records)}개")

    # 데이터셋 생성
    if qa_records:
        korquad_formatted = Dataset.from_list(qa_records)
        print(f"KorQuAD 형식 변환 완료: {len(korquad_formatted)} 샘플")

        # 데이터셋 병합
        print("\nKoCommercial과 KorQuAD 데이터셋 병합 중...")
        combined_dataset = concatenate_datasets([ko_commercial_formatted, korquad_formatted])
        print(f"병합된 데이터셋 크기: {len(combined_dataset)} 샘플")
    else:
        print("KorQuAD에서 유효한 샘플을 추출할 수 없어 KoCommercial만 사용합니다.")
        combined_dataset = ko_commercial_formatted

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
    num_train_epochs=1,  # 병합된 데이터셋은 크기가 크므로 1 에포크
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