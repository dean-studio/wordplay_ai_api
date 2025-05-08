# wordplay_ai_api
wordplay_ai_api22

## redoc
http://34.53.35.1:8000/redoc

35.233.152.5:8283

35.233.152.5:7860

35.233.152.5:5000 /tensor board

## venv
https://huggingface.co/datasets/MarkrAI/KoCommercial-Dataset

https://huggingface.co/datasets/KorQuAD/squad_kor_v2

## venv

```
# 1. venv 만들기
python3 -m venv venv

# 2. venv 활성화
source venv/bin/activate

# 3. 이 상태에서 설치
pip install -r requirements.txt

```


## nvidia 

```
# 1. NVIDIA repo 키 등록
sudo apt update
sudo apt install -y wget gnupg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 2. 드라이버 목록 갱신
sudo apt update

# 3. 드라이버 설치 (L4는 최신 535 이상 사용 권장)
sudo apt install -y nvidia-driver-535

# 4. 설치 완료 후 재부팅
sudo reboot

```

## hugginface
```
huggingface-cli login
huggingface-cli snapshot-download beomi/KoAlpaca-Polyglot-5.8B --local-dir ./models/koalpaca-5.8b --local-dir-use-syml

```