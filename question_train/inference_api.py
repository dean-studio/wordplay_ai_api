# Filename: infer_test.py

# 환경 변수 설정: 메모리 파편화 문제 완화를 위해
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import uvicorn
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# peft 라이브러리 임포트 (LoRA 어댑터 로드를 위해)
from peft import PeftModel

# 로그 설정: DEBUG 레벨
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KoAlpaca Inference & Q/A API",
    description="koAlpaca-Polyglot-5.8B 모델에 LoRA 어댑터를 추가로 로드하여 텍스트 생성 및 질문 응답하는 API입니다."
)

MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"사용할 디바이스: {device}")

try:
    logger.info(f"모델 {MODEL_NAME} 로딩 시작")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        load_in_8bit=True,  # 8-bit 양자화로 메모리 사용량 최적화 (bitsandbytes 라이브러리 필요)
        device_map="auto"  # 사용 가능한 디바이스에 자동 할당 (GPU/CPU 혼용)
    )
    logger.info("모델과 토크나이저 로딩 완료")
except Exception as e:
    logger.exception("모델 로딩 중 에러 발생")
    raise RuntimeError(f"모델 로딩 중 에러 발생: {e}")

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
        tokens.pop("token_type_ids", None)
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
