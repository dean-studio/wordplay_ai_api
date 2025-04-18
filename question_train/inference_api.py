# Filename: infer_test.py

# 환경 변수 설정: 메모리 파편화 문제 완화를 위해
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import uvicorn
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# peft 라이브러리 임포트 (LoRA 어댑터 로드를 위해)
from peft import PeftModel, prepare_model_for_kbit_training
# 로그 설정: DEBUG 레벨
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KoAlpaca Inference & Q/A API",
    description="koAlpaca-Polyglot-5.8B 모델에 LoRA 어댑터를 추가로 로드하여 텍스트 생성 및 질문 응답하는 API입니다."
)

# 모델 관련 상수
MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
LOCAL_PATH = "./models/koalpaca-5.8b"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"사용할 디바이스: {device}")

# 로컬 경로에 모델이 있다면 해당 경로에서 불러오기 (훈련 시 사용했던 동일 조건)
try:
    logger.info(f"로컬 모델 경로({LOCAL_PATH})에서 모델 로드 시작")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
    # 4-bit 양자화 설정 (train.py와 동일)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )
    logger.info("로컬 모델 로드 완료 (4-bit 양자화, 자동 device mapping).")

    # train.py와 동일하게 k-bit 훈련에 맞게 모델 준비
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    logger.info("모델이 k-bit 훈련에 맞게 준비됨 (캐시 비활성화, gradient checkpointing 활성화).")
except Exception as e:
    logger.exception("모델 로딩 중 에러 발생")
    raise RuntimeError(f"모델 로딩 중 에러 발생: {e}")

# 로컬 모델에 LoRA 어댑터 로드 (튜닝 시와 동일한 어댑터 체크포인트 사용)
try:
    lora_adapter_path = "./output/koalpaca-lora"
    logger.info("LoRA 어댑터 로딩 시작")
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    logger.info("LoRA 어댑터 로딩 완료")
except Exception as e:
    logger.exception("LoRA 어댑터 로딩 중 에러 발생")
    raise RuntimeError(f"LoRA 어댑터 로딩 중 에러 발생: {e}")


# [텍스트 생성] 요청 Body 모델
class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 200

# [질문 응답] 요청 Body 모델
class QuestionRequest(BaseModel):
    question: str
    max_length: int = 200

@app.post("/infer", summary="텍스트 생성", description="주어진 프롬프트 기반으로 텍스트를 생성합니다.")
async def infer(request: InferenceRequest):
    try:
        logger.debug(f"[infer] 요청 수신: {request}")
        tokens = tokenizer(request.prompt, return_tensors="pt")
        tokens.pop("token_type_ids", None)  # 사용하지 않는 token_type_ids 제거
        # 모델이 자동으로 할당된 디바이스로 토큰 전송
        tokens = {k: v.to(device) for k, v in tokens.items()}
        logger.debug(f"[infer] 토큰을 {device}로 전송 완료")

        start_time = time.time()
        outputs = model.generate(
            **tokens,
            max_length=request.max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        elapsed = time.time() - start_time
        logger.debug(f"[infer] 모델 추론 완료, 소요 시간: {elapsed:.2f}초")

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"[infer] 디코딩 결과: {result}")
        return {"result": result}
    except Exception as e:
        logger.exception("[infer] 추론 중 에러 발생")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/ask", summary="질문 응답", description="사용자의 질문에 대해 모델이 답변을 생성합니다.")
async def ask(request: QuestionRequest):
    try:
        logger.debug(f"[ask] 요청 수신: {request}")
        prompt_text = f"질문: {request.question}\n답변:"
        tokens = tokenizer(prompt_text, return_tensors="pt")
        tokens.pop("token_type_ids", None)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        logger.debug(f"[ask] 토큰을 {device}로 전송 완료")

        start_time = time.time()
        outputs = model.generate(
            **tokens,
            max_length=request.max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        elapsed = time.time() - start_time
        logger.debug(f"[ask] 모델 추론 완료, 소요 시간: {elapsed:.2f}초")

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("답변:")[-1].strip()
        logger.debug(f"[ask] 추출된 답변: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.exception("[ask] 질문 응답 처리 중 에러 발생")
        raise HTTPException(status_code=500, detail=f"Question answering error: {str(e)}")

if __name__ == "__main__":
    # reload 옵션은 중복 로드를 방지하기 위해 False로 실행합니다.
    uvicorn.run("infer_test:app", host="0.0.0.0", port=8000, reload=False)
