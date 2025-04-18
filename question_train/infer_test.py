# Filename: infer_test.py

import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 로그 설정: DEBUG 레벨로 모든 로그 출력
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KoAlpaca Inference & Q/A API",
    description="beomi/KoAlpaca-Polyglot-5.8B 모델을 사용한 텍스트 생성 및 질문 응답 API입니다."
)

MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
try:
    logger.info(f"모델 {MODEL_NAME} 로딩 시작")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logger.info("모델과 토크나이저 로딩 완료")
except Exception as e:
    logger.exception("모델 로딩 중 에러 발생")
    raise RuntimeError(f"모델 로딩 중 에러 발생: {e}")


# [텍스트 생성] 요청 Body 모델
class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 200  # 생성을 위한 최대 길이


# [질문 응답] 요청 Body 모델
class QuestionRequest(BaseModel):
    question: str
    max_length: int = 200  # 응답 텍스트의 최대 길이


@app.post("/infer", summary="텍스트 생성", description="주어진 프롬프트를 기반으로 텍스트를 생성합니다.")
async def infer(request: InferenceRequest):
    try:
        logger.debug(f"[infer] 요청 수신: {request}")
        tokens = tokenizer(request.prompt, return_tensors="pt")
        logger.debug(f"[infer] 토큰 생성 결과: {tokens}")
        tokens.pop("token_type_ids", None)  # 사용하지 않는 파라미터 제거
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        logger.debug("[infer] 토큰을 모델 디바이스로 전송 완료")

        outputs = model.generate(
            **tokens,
            max_length=request.max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        logger.debug("[infer] 모델 추론 완료")
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
        # 질문 응답용 프롬프트 구성
        prompt_text = f"질문: {request.question}\n답변:"
        logger.debug(f"[ask] 구성된 프롬프트: {prompt_text}")

        tokens = tokenizer(prompt_text, return_tensors="pt")
        logger.debug(f"[ask] 토큰 생성 결과: {tokens}")
        tokens.pop("token_type_ids", None)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        logger.debug("[ask] 토큰을 모델 디바이스로 전송 완료")

        outputs = model.generate(
            **tokens,
            max_length=request.max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        logger.debug("[ask] 모델 추론 완료")
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
