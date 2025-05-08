from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import Trainer
import os
import json
import torch
import gc
import psutil  # 시스템 메모리 모니터링용


# 메모리 사용량 보고 함수
def report_memory():
    process = psutil.Process(os.getpid())
    print(f"메모리 사용량: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU 메모리 할당량: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
        print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")


# 메모리 초기화
gc.collect()
torch.cuda.empty_cache()
report_memory()

# TensorBoard 로그 디렉토리 생성
tensorboard_dir = "./tensorboard_logs"
os.makedirs(tensorboard_dir, exist_ok=True)

# 모델 및 토크나이저 로드 (더 낮은 비트 수 사용)
print("모델 로딩 중...")
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

# 모델을 필요한 시점까지 로드하지 않음
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 특수 토큰 설정 확인
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 데이터셋 준비 함수 - KoCommercial
def prepare_commercial_sample(example):
    try:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        # 프롬프트 생성
        user_content = instruction
        if input_text:
            user_content += f"\n{input_text}"

        # CLOVA 형식으로 변환
        chat = [
            {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]

        # 챗 템플릿 적용
        full_text = tokenizer.apply_chat_template(chat, tokenize=False)
        return {"text": full_text}
    except:
        return None


# 데이터셋 준비 함수 - KorQuAD
def prepare_korquad_sample(example):
    try:
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
    except:
        return None


# 데이터 처리 및 학습을 나눠서 진행
data_path = "./processed_data"
os.makedirs(data_path, exist_ok=True)

# 단계 1: KoCommercial 데이터셋 처리 (이전에 처리했는지 확인)
commercial_processed_path = os.path.join(data_path, "commercial_processed.json")
if os.path.exists(commercial_processed_path):
    print(f"기존 처리된 KoCommercial 데이터 로드 중: {commercial_processed_path}")
    with open(commercial_processed_path, 'r', encoding='utf-8') as f:
        commercial_count = int(f.readline().strip())
    print(f"KoCommercial 처리된 샘플 수: {commercial_count}")
else:
    print("KoCommercial 데이터셋 로딩 및 처리 중...")
    ko_commercial = load_dataset("MarkrAI/KoCommercial-Dataset")

    # 부분 처리를 위한 KoCommercial 데이터 수
    sample_count = min(100000, len(ko_commercial["train"]))  # 10만 샘플로 제한

    commercial_samples = []
    for i in range(sample_count):
        if i % 10000 == 0:
            print(f"KoCommercial 처리 중: {i}/{sample_count}")
            report_memory()
            gc.collect()

        sample = ko_commercial["train"][i]
        processed = prepare_commercial_sample(sample)
        if processed is not None:
            commercial_samples.append(processed)

    commercial_count = len(commercial_samples)
    print(f"처리된 KoCommercial 샘플 수: {commercial_count}")

    # 처리된 샘플 수만 저장 (메모리 절약)
    with open(commercial_processed_path, 'w', encoding='utf-8') as f:
        f.write(str(commercial_count))

    # 메모리 정리
    del ko_commercial
    del commercial_samples
    gc.collect()
    torch.cuda.empty_cache()
    report_memory()

# 단계 2: KorQuAD 데이터셋 처리 (이전에 처리했는지 확인)
korquad_processed_path = os.path.join(data_path, "korquad_processed.json")
if os.path.exists(korquad_processed_path):
    print(f"기존 처리된 KorQuAD 데이터 로드 중: {korquad_processed_path}")
    with open(korquad_processed_path, 'r', encoding='utf-8') as f:
        korquad_count = int(f.readline().strip())
    print(f"KorQuAD 처리된 샘플 수: {korquad_count}")
else:
    print("KorQuAD 데이터셋 로딩 및 처리 중...")
    try:
        korquad_v2 = load_dataset("KorQuAD/squad_kor_v2", trust_remote_code=True)

        # 부분 처리를 위한 KorQuAD 데이터 수
        sample_count = min(50000, len(korquad_v2["train"]))  # 5만 샘플로 제한

        korquad_samples = []
        for i in range(sample_count):
            if i % 5000 == 0:
                print(f"KorQuAD 처리 중: {i}/{sample_count}")
                report_memory()
                gc.collect()

            sample = korquad_v2["train"][i]
            processed = prepare_korquad_sample(sample)
            if processed is not None:
                korquad_samples.append(processed)

        korquad_count = len(korquad_samples)
        print(f"처리된 KorQuAD 샘플 수: {korquad_count}")

        # 처리된 샘플 수만 저장 (메모리 절약)
        with open(korquad_processed_path, 'w', encoding='utf-8') as f:
            f.write(str(korquad_count))

        # 메모리 정리
        del korquad_v2
        del korquad_samples
        gc.collect()
        torch.cuda.empty_cache()
        report_memory()
    except Exception as e:
        print(f"KorQuAD 처리 중 오류 발생: {str(e)}")
        korquad_count = 0

# 단계 3: 이제 모델 로드 (데이터 처리 완료 후)
print("모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)

# LoRA 설정
lora_config = LoraConfig(
    r=4,  # 더 작은 rank 값
    lora_alpha=8,  # 더 작은 alpha 값
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# LoRA 모델 생성
peft_model = get_peft_model(model, lora_config)
print("LoRA 모델 초기화 완료")
report_memory()

# 단계 4: 데이터 토큰화 및 학습 준비
print("\n데이터셋 생성 및 토큰화 준비 중...")

# 총 샘플 수
total_samples = commercial_count + korquad_count
print(f"총 샘플 수: {total_samples}")

# 청크 단위로 처리 (메모리 사용량 관리)
chunk_size = 10000
num_chunks = (total_samples + chunk_size - 1) // chunk_size


# 토큰화 함수
def tokenize_text(text):
    result = tokenizer(
        text,
        truncation=True,
        max_length=768,  # 더 짧은 시퀀스 길이
        padding="max_length",
        return_tensors="pt"
    )
    result["labels"] = result["input_ids"].clone()
    return {
        "input_ids": result["input_ids"][0],
        "attention_mask": result["attention_mask"][0],
        "labels": result["labels"][0]
    }


# 청크별 데이터 처리 및 학습
for chunk_idx in range(num_chunks):
    print(f"\n청크 {chunk_idx + 1}/{num_chunks} 처리 중...")

    # 청크 데이터셋 생성 경로
    chunk_path = os.path.join(data_path, f"chunk_{chunk_idx}.pt")

    if os.path.exists(chunk_path):
        print(f"기존 처리된 청크 로드 중: {chunk_path}")
        # 청크 데이터는 토큰화 이후 단계에서 로드
    else:
        # 이 청크에 필요한 샘플 수 계산
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        current_chunk_size = end_idx - start_idx

        print(f"청크 {chunk_idx + 1} 데이터 생성 중: {start_idx}~{end_idx - 1}")

        # 샘플 생성
        chunk_samples = []

        # 상업 데이터 처리
        commercial_end = min(commercial_count, end_idx)
        if start_idx < commercial_count:
            commercial_start = start_idx
            print(f"KoCommercial 샘플 처리: {commercial_start}~{commercial_end - 1}")

            ko_commercial = load_dataset("MarkrAI/KoCommercial-Dataset")
            for i in range(commercial_start, commercial_end):
                if (i - commercial_start) % 1000 == 0:
                    print(f"KoCommercial 처리 중: {i - commercial_start}/{commercial_end - commercial_start}")

                sample = ko_commercial["train"][i]
                processed = prepare_commercial_sample(sample)
                if processed is not None:
                    tokenized = tokenize_text(processed["text"])
                    chunk_samples.append(tokenized)

            # 메모리 정리
            del ko_commercial
            gc.collect()

        # KorQuad 데이터 처리
        if commercial_count < end_idx and korquad_count > 0:
            korquad_start = max(0, start_idx - commercial_count)
            korquad_end = min(korquad_count, end_idx - commercial_count)

            if korquad_start < korquad_end:
                print(f"KorQuAD 샘플 처리: {korquad_start}~{korquad_end - 1}")

                korquad_v2 = load_dataset("KorQuAD/squad_kor_v2", trust_remote_code=True)
                for i in range(korquad_start, korquad_end):
                    if (i - korquad_start) % 1000 == 0:
                        print(f"KorQuAD 처리 중: {i - korquad_start}/{korquad_end - korquad_start}")

                    sample = korquad_v2["train"][i]
                    processed = prepare_korquad_sample(sample)
                    if processed is not None:
                        tokenized = tokenize_text(processed["text"])
                        chunk_samples.append(tokenized)

                # 메모리 정리
                del korquad_v2
                gc.collect()

        # 청크 데이터 저장
        print(f"청크 {chunk_idx + 1} 데이터 저장 중: {len(chunk_samples)} 샘플")
        torch.save(chunk_samples, chunk_path)

        # 메모리 정리
        del chunk_samples
        gc.collect()
        torch.cuda.empty_cache()

    # 청크 데이터 로드 및 학습
    print(f"청크 {chunk_idx + 1} 학습 시작...")

    # 데이터 로드
    chunk_samples = torch.load(chunk_path)

    # 데이터셋 생성
    features = ['input_ids', 'attention_mask', 'labels']
    chunk_dataset = Dataset.from_dict({
        feature: [sample[feature] for sample in chunk_samples]
        for feature in features
    })

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=f"./clova-lora-chunk{chunk_idx}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        save_steps=500,
        logging_dir=tensorboard_dir,
        logging_steps=50,
        report_to=["tensorboard"],
        fp16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        save_total_limit=1,
        save_strategy="steps",
        remove_unused_columns=False,
    )

    # Trainer 초기화
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=chunk_dataset,
    )

    # 청크 학습
    trainer.train()

    # 체크포인트 저장
    checkpoint_path = f"./clova-lora-checkpoint-chunk{chunk_idx}"
    peft_model.save_pretrained(checkpoint_path)
    print(f"청크 {chunk_idx + 1} 체크포인트 저장 완료: {checkpoint_path}")

    # 메모리 정리
    del chunk_samples
    del chunk_dataset
    gc.collect()
    torch.cuda.empty_cache()
    report_memory()

# 최종 모델 저장
final_model_path = "./clova-lora-final"
peft_model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\n최종 모델 저장 완료: {final_model_path}")
print("\n모든 과정이 완료되었습니다!")