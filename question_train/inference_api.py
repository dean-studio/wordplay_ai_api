from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re
import json

app = FastAPI()

base_model = "./models/koalpaca-5.8b"
adapter_path = "./output/koalpaca-lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

class QuestionRequest(BaseModel):
    question: str

@app.post("/infer")
async def infer(req: QuestionRequest):
    prompt = f"""
다음 질문을 바탕으로 다음과 같은 형식으로 문제를 생성해줘:

1. 객관식 문제
- 질문: ?
- 보기: 1. 보기1 / 2. 보기2 / 3. 보기3 / 4. 보기4
- 정답: 번호
- 해설: ...

2. OX 문제
- 질문: ?
- 정답: O 또는 X
- 해설: ...

3. 주관식 해설
- 설명: 해당 질문에 대한 사실 기반 설명

질문: {req.question}
    """.strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = output.replace(prompt, "").strip()

    # 간단한 파싱
    mc_question = re.search(r"객관식 문제\s*- 질문: (.*?)\n", result)
    options = re.findall(r"\d+\.\s*(.*?)\s*/?", result)
    answer = re.search(r"정답: (\d+)", result)
    explanation = re.search(r"해설: (.*?)\n", result)

    ox_q = re.search(r"OX 문제\s*- 질문: (.*?)\n", result)
    ox_a = re.search(r"정답: ([OX])", result)
    ox_exp = re.search(r"OX 문제.*?해설: (.*?)\n", result, re.DOTALL)

    subjective = re.search(r"설명: (.+)", result)

    # 간단한 분류
    category = "색깔"
    category_idx = 102
    grade_level = "초등 1학년"
    difficulty = 1

    response = {
        "question": req.question,
        "category": category,
        "category_idx": category_idx,
        "grade_level": grade_level,
        "difficulty": difficulty,
        "multiple_choice": {
            "question": mc_question.group(1) if mc_question else None,
            "options": [{"id": i+1, "value": v.strip()} for i, v in enumerate(options)],
            "answer": int(answer.group(1)) if answer else None,
            "explanation": explanation.group(1) if explanation else subjective.group(1) if subjective else ""
        },
        "ox_quiz": {
            "question": ox_q.group(1) if ox_q else None,
            "answer": True if ox_a and ox_a.group(1) == "O" else False,
            "explanation": ox_exp.group(1) if ox_exp else ""
        },
        "reference_links": {
            "priority_1": None,
            "priority_2": None,
            "priority_3": None
        }
    }

    return response