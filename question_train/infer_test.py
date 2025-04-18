# Filename: infer_test.py

# 환경 변수 설정: 파편화 문제 완화를 위해
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import uvicorn
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 로그 설정: DEBUG 레벨
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KoAlpaca Inference & Q/A API",
    description="beomi/KoAlpaca-Polyglot-5.8B 모델을 GPU에 강제로 로드하여 텍스트 생성 및 질문 응답 API입니다."
)

MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"사용할 디바이스: {device}")

try:
    logger.info(f"모델 {MODEL_NAME} 로딩 시작 (강제 GPU 할당: device_map={{'': '{device}'}})")
    # 모델을 GPU에 강제로 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": device}  # 모든 파라미터를 지정된 GPU에 할당
    )
    logger.info("모델과 토크나이저 로딩 완료")
except Exception as e:
    logger.exception("모델 강제 GPU 로딩 중 에러 발생")
    # 에러 메시지 안내: 만약 여기서 OOM 오류가 난다면, 강제 GPU 할당 대신 device_map='auto' 사용을 고려하거나 load_in_8bit 등을 검토하세요.
    raise RuntimeError(f"모델 로딩 중 에러 발생: {e}")

# 디버깅: 모델의 모든 파라미터가 실제로 어느 디바이스에 있는지 출력 (GPU 할당 여부 확인)
for name, param in model.named_parameters():
    logger.debug(f"{name}: {param.device}")


# [텍스트 생성] 요청 Body 모델
class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 200  # 필요에 따라 생성 최대 길이 조정


# [질문 응답] 요청 Body 모델
class QuestionRequest(BaseModel):
    question: str
    max_length: int = 200


@app.post("/infer", summary="텍스트 생성", description="주어진 프롬프트 기반으로 텍스트를 생성합니다.")
async def infer(request: InferenceRequest):
    try:
        logger.debug(f"[infer] 요청 수신: {request}")
        tokens = tokenizer(request.prompt, return_tensors="pt")
        logger.debug(f"[infer] 토큰 생성 결과: {tokens}")
        tokens.pop("token_type_ids", None)  # 사용하지 않는 파라미터 제거
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
        logger.debug(f"[ask] 구성된 프롬프트: {prompt_text}")

        tokens = tokenizer(prompt_text, return_tensors="pt")
        logger.debug(f"[ask] 토큰 생성 결과: {tokens}")
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
        logger.debug(f"[ask] 생성된 텍스트: {generated_text}")
        answer = generated_text.split("답변:")[-1].strip()
        logger.debug(f"[ask] 추출된 답변: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.exception("[ask] 질문 응답 처리 중 에러 발생")
        raise HTTPException(status_code=500, detail=f"Question answering error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("infer_test:app", host="0.0.0.0", port=8000, reload=True)
