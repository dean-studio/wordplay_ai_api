import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from typing import List, Dict, Any


class ModelManager:
    """모델 로드 및 관리 클래스"""

    def __init__(self, model_id: str, lora_paths: List[str]):
        self.model_id = model_id
        self.lora_paths = lora_paths
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        print(f"기본 모델 '{self.model_id}' 로드 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # LoRA 어댑터 적용
        for lora_path in self.lora_paths:
            if os.path.exists(lora_path):
                print(f"LoRA 어댑터 로드 중: {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                print(f"LoRA 어댑터 {os.path.basename(lora_path)}가 성공적으로 적용되었습니다.")
            else:
                print(f"경고: LoRA 어댑터 경로({lora_path})를 찾을 수 없습니다.")

        self.model.eval()
        print(f"사용 장치: {self.device}")

    def generate_text(self, system_message: str, user_message: str,
                      max_new_tokens: int = 1024, temperature: float = 0.7,
                      top_p: float = 0.9, repetition_penalty: float = 1.2) -> str:
        """텍스트 생성 기본 함수"""
        chat = [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        inputs = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_strings=["<|endofturn|>", "<|stop|>"],
                tokenizer=self.tokenizer
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        full_output = "\n".join(output)

        if "assistant" in full_output:
            result = full_output.split("assistant", 1)[1].strip()
            return result
        else:
            return full_output