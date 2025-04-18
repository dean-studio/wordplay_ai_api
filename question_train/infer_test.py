# Filename: infer_test.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(
    title="KoAlpaca Inference & Q/A API",
    description="beomi/KoAlpaca-Polyglot-5.8B 모델을 사용한 텍스트 생성 및 질문 응답 API 입니다."
)

MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # GPU 사용 시 float16 권장
        device_map="auto"
    )
except Exception as e:
    raise RuntimeError(f"모델 로딩 중 에러 발생: {e}")

# [텍스트 생성] 요청 Body 모델
class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 200  # 생성 최대 길이 (필요에 따라 조정)

# [질문 응답] 요청 Body 모델
class QuestionRequest(BaseModel):
    question: str
    max_length: int = 200  # 답변 텍스트의 최대 길이 (필요에 따라 조정)

@app.post("/infer", summary="텍스트 생성", description="주어진 프롬프트 기반으로 텍스트를 생성합니다.")
async def infer(request: InferenceRequest):
    try:
        # 토크나이저가 반환하는 딕셔너리에서 token_type_ids 제거 (모델에서 사용하지 않음)
        tokens = tokenizer(request.prompt, return_tensors="pt")
        tokens.pop("token_type_ids", None)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        outputs = model.generate(
            **tokens,
            max_length=request.max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/ask", summary="질문 응답", description="사용자의 질문에 대해 모델이 자동으로 답변을 생성합니다.")
async def ask(request: QuestionRequest):
    try:
        # 질문 응답용 프롬프트 구성: '질문:' 이후 '답변:' 프롬프트 추가
        prompt_text = f"질문: {request.question}\n답변:"
        tokens = tokenizer(prompt_text, return_tensors="pt")
        tokens.pop("token_type_ids", None)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        outputs = model.generate(
            **tokens,
            max_length=request.max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # '답변:' 이후의 텍스트를 추출하여 리턴
        answer = generated_text.split("답변:")[-1].strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("infer_test:app", host="0.0.0.0", port=8000, reload=True)
