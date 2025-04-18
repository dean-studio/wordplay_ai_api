import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="KoAlpaca Inference API")

tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B", device_map="auto")

# 요청 Body 모델 정의
class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 150  # 최대 생성 길이 (필요에 따라 변경)

@app.post("/infer")
def infer(request: InferenceRequest):
    # 입력문을 토크나이즈
    inputs = tokenizer(request.prompt, return_tensors="pt")
    # 텍스트 생성. 필요시 do_sample, temperature 등 파라미터 조정 가능
    outputs = model.generate(
        **inputs,
        max_length=request.max_length,
        do_sample=True,      # 샘플링 사용 여부
        temperature=0.7,     # 생성 온도
        top_k=50,            # 상위 k 단어 내에서 샘플링
        top_p=0.95,          # 누적 확률 p 내에서 샘플링
    )
    # 생성된 토큰을 텍스트로 변환
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": generated_text}

if __name__ == "__main__":
    # uvicorn을 통해 앱 실행 (개발모드: reload 활성화)
    uvicorn.run("infer_test:app", host="0.0.0.0", port=8000, reload=True)
