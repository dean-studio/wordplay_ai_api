from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets, Features, Value
from trl import SFTTrainer
import os
import json

# TensorBoard 로그 디렉토리 생성
tensorboard_dir = "./tensorboard_logs"
os.makedirs(tensorboard_dir, exist_ok=True)

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
print("KoCommercial-Dataset 로딩 중...")
ko_commercial = load_dataset("MarkrAI/KoCommercial-Dataset")

# 데이터셋 구조 확인
print("KoCommercial 샘플 형식 확인:")
ko_sample = ko_commercial["train"][0]
print(f"키: {list(ko_sample.keys())}")
for key, value in ko_sample.items():
    print(f"- {key}: {type(value)}")

# KoQuAD 로드 및 처리
try:
    print("\nKorQuAD/squad_kor_v2 로딩 중...")
    korquad_v2 = load_dataset("KorQuAD/squad_kor_v2", trust_remote_code=True)

    # KorQuAD 구조 확인
    print("\nKorQuAD 샘플 형식 확인:")
    kq_sample = korquad_v2["train"][0]
    print(f"키: {list(kq_sample.keys())}")
    for key, value in kq_sample.items():
        print(f"- {key}: {type(value)}")
        if hasattr(value, "keys"):
            print(f"  하위 키: {list(value.keys())}")


    # KorQuAD 데이터를 일관된 형식으로 변환
    def convert_korquad_to_standard_format(example):
        try:
            instruction = f"다음 문서를 읽고 질문에 답하세요.\n\n문서: {example['context']}\n\n질문: {example['question']}"

            # 답변 텍스트 추출
            answer_text = ""
            if "answers" in example:
                if isinstance(example["answers"], dict) and "text" in example["answers"]:
                    texts = example["answers"]["text"]
                    answer_text = texts[0] if isinstance(texts, list) and texts else ""

            # 정확히 같은 필드 구조로 반환
            return {
                "instruction": instruction,
                "input": "",
                "output": answer_text if answer_text else "답변을 찾을 수 없습니다."
            }
        except Exception as e:
            print(f"변환 중 오류: {str(e)}")
            return {
                "instruction": "오류가 발생했습니다.",
                "input": "",
                "output": "데이터 처리 중 오류가 발생했습니다."
            }


    # 테스트 변환
    print("\n샘플 변환 테스트 중...")
    test_samples = korquad_v2["train"].select(range(5))
    converted_test = test_samples.map(convert_korquad_to_standard_format)

    # 변환 결과 확인
    print("\n변환 결과 샘플:")
    print(json.dumps(converted_test[0], indent=2, ensure_ascii=False))

    # 전체 데이터셋 변환 - 필드 구조 강제 지정
    print("\n전체 KorQuAD 데이터셋 변환 중...")
    features = Features({
        'instruction': Value('string'),
        'input': Value('string'),
        'output': Value('string')  # 문자열로 강제 변환
    })

    # 명시적인 필드 구조로 변환
    korquad_v2_converted = korquad_v2.map(
        convert_korquad_to_standard_format,
        features=features  # 명시적 특성 지정
    )

    # 구조 확인
    print("\n변환 후 KorQuAD 샘플 형식 확인:")
    kq_conv_sample = korquad_v2_converted["train"][0]
    print(f"키: {list(kq_conv_sample.keys())}")
    for key, value in kq_conv_sample.items():
        print(f"- {key}: {type(value)}")

    # KoCommercial 데이터셋도 동일한 특성으로 변환
    print("\nKoCommercial 데이터셋 필드 구조 통일 중...")
    ko_commercial_standardized = ko_commercial["train"].cast(features)

    # 데이터셋 병합
    print("\n데이터셋 병합 중...")
    combined_dataset = concatenate_datasets([ko_commercial_standardized, korquad_v2_converted["train"]])
    print(f"병합된 데이터셋 크기: {len(combined_dataset)} 샘플")

except Exception as e:
    print(f"\nKorQuAD 처리 중 오류 발생: {str(e)}")
    print("KoCommercial-Dataset만 사용합니다.")

    # KoCommercial 데이터셋만 사용
    features = Features({
        'instruction': Value('string'),
        'input': Value('string'),
        'output': Value('string')
    })
    combined_dataset = ko_commercial["train"].cast(features)


# 형식에 맞게 변환
def format_for_clova(example):
    chat = [
        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
        {"role": "user",
         "content": example["instruction"] + (f"\n{example['input']}" if example.get("input", "") else "")},
        {"role": "assistant", "content": example.get("output", "")}
    ]
    return {"text": tokenizer.apply_chat_template(chat, tokenize=False)}


print("\n데이터를 CLOVA 형식으로 변환 중...")
formatted_dataset = combined_dataset.map(format_for_clova)

# 학습 설정
print("\n학습 설정 구성 중...")
training_args = TrainingArguments(
    output_dir="./clova-lora-with-korquad",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=15000,
    save_steps=1000,

    # TensorBoard 설정
    logging_dir=tensorboard_dir,
    logging_strategy="steps",
    logging_steps=10,
    report_to=["tensorboard"],

    # 기존 설정
    fp16=True,
    optim="paged_adamw_8bit",
)

# 트레이너 설정
print("\n트레이너 초기화 중...")
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_text_field="text",
)

# 학습 시작
print("\n학습 시작 중...")
print(f"데이터셋 크기: {len(formatted_dataset)}")
print(f"TensorBoard 로그 위치: {tensorboard_dir}")
print(f"학습 결과 저장 위치: ./clova-lora-with-korquad")
trainer.train()

# 모델 저장
print("\n모델 저장 중...")
peft_model.save_pretrained("./clova-lora-korquad-final")
print("\n모든 과정이 완료되었습니다!")