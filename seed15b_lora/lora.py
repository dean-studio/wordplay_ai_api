from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
import os

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
print("데이터셋 로딩 중...")
ko_commercial = load_dataset("MarkrAI/KoCommercial-Dataset")
korquad_v2 = load_dataset("KorQuAD/squad_kor_v2", trust_remote_code=True)

# KorQuAD 데이터셋 구조 확인
print("\nKorQuAD 데이터셋 구조 확인:")
print("사용 가능한 스플릿:", list(korquad_v2.keys()))
print("샘플 데이터 구조:")
sample = korquad_v2["train"][0]
for key, value in sample.items():
    print(f"- {key}: {type(value)}")
    if hasattr(value, "keys"):
        print(f"  하위 키: {list(value.keys())}")
    elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], "keys"):
        print(f"  리스트 내 첫 항목 키: {list(value[0].keys())}")


# KorQuAD 데이터셋 형식 변환 함수 (구조에 맞게 수정)
def convert_korquad_format(example):
    # 데이터셋 구조에 따라 적절히 수정
    try:
        # 일반적인 SQuAD 형식 시도
        if "context" in example and "question" in example:
            instruction = f"다음 문서를 읽고 질문에 답하세요.\n\n문서: {example['context']}\n\n질문: {example['question']}"

            # answers 구조 확인 및 처리
            if "answers" in example:
                if isinstance(example["answers"], dict) and "text" in example["answers"]:
                    answer_texts = example["answers"]["text"]
                    answer_text = answer_texts[0] if isinstance(answer_texts,
                                                                list) and answer_texts else "답변을 찾을 수 없습니다."
                elif isinstance(example["answers"], list) and example["answers"]:
                    answer_text = example["answers"][0]["text"] if "text" in example["answers"][0] else "답변을 찾을 수 없습니다."
                else:
                    answer_text = "답변 구조를 인식할 수 없습니다."
            elif "answer" in example:
                # 일부 데이터셋은 'answer' 키를 사용
                answer_text = example["answer"]
            else:
                answer_text = "답변을 찾을 수 없습니다."

            return {
                "instruction": instruction,
                "input": "",
                "output": answer_text
            }
        else:
            print(f"예상치 못한 데이터 구조: {list(example.keys())}")
            return {
                "instruction": "질문에 답하세요.",
                "input": "",
                "output": "데이터 구조를 인식할 수 없습니다."
            }
    except Exception as e:
        print(f"데이터 변환 중 오류 발생: {e}")
        return {
            "instruction": "질문에 답하세요.",
            "input": "",
            "output": "오류가 발생했습니다."
        }


# 일부 샘플만 변환하여 테스트
print("\n소수의 샘플로 변환 테스트 중...")
test_samples = korquad_v2["train"].select(range(min(5, len(korquad_v2["train"]))))
converted_test = test_samples.map(convert_korquad_format)
print("변환 테스트 완료. 첫 번째 변환 결과:")
print(converted_test[0])

# 테스트가 성공하면 전체 데이터셋 변환
print("\nKorQuAD 전체 데이터 변환 중...")
korquad_v2_converted = korquad_v2.map(convert_korquad_format)

# 두 데이터셋 합치기 (필요한 필드만 선택)
print("데이터셋 병합 중...")
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


print("데이터 형식 변환 중...")
formatted_dataset = combined_dataset.map(format_for_clova)

# 학습 설정 - TensorBoard 로깅 추가
print("학습 설정 구성 중...")
training_args = TrainingArguments(
    output_dir="./clova-lora-with-korquad",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=15000,
    save_steps=1000,

    # TensorBoard 로깅 설정
    logging_dir=tensorboard_dir,
    logging_strategy="steps",
    logging_steps=10,
    report_to=["tensorboard"],

    # 기존 설정
    fp16=True,
    optim="paged_adamw_8bit",
)

# 트레이너 설정
print("트레이너 초기화 중...")
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_text_field="text",
)

# 학습 시작
print("학습 시작, TensorBoard 로그가 생성됩니다...")
trainer.train()

# 모델 저장
print("학습 완료, 모델 저장 중...")
peft_model.save_pretrained("./clova-lora-korquad-final")
print("모든 과정이 완료되었습니다!")