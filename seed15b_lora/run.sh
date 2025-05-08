#!/bin/bash

# 파일명: run.sh
# 사용법: bash run.sh

# 로그 디렉토리 생성
TENSORBOARD_LOG_DIR="./tensorboard_logs"
mkdir -p $TENSORBOARD_LOG_DIR

# 학습 로그 디렉토리 생성
mkdir -p ./logs

# 현재 시간을 파일명으로 사용
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/training_$TIMESTAMP.log"

echo "Starting TensorBoard on port 5000..."

tensorboard --logdir=$TENSORBOARD_LOG_DIR --host=0.0.0.0 --port=5000 > ./logs/tensorboard_$TIMESTAMP.log 2>&1 &
TENSORBOARD_PID=$!

echo "TensorBoard started with PID: $TENSORBOARD_PID"
echo "TensorBoard available at http://localhost:5000"
echo "If running on a remote server, access via: http://SERVER_IP:5000"

echo "Starting LoRA training..."
echo "Training logs will be saved to: $LOG_FILE"

# 학습 스크립트 실행 및 로그 저장
python lora.py > $LOG_FILE 2>&1
TRAINING_EXIT_CODE=$?

echo "Training finished with exit code: $TRAINING_EXIT_CODE"

# 학습이 끝나도 TensorBoard는 계속 실행
echo "TensorBoard is still running with PID: $TENSORBOARD_PID"
echo "To stop TensorBoard manually, run: kill $TENSORBOARD_PID"

# 학습 결과에 따른 메시지 출력
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Trained model saved to ./clova-lora-final"
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check the logs at $LOG_FILE for details"
fi

echo "To review training logs, run: cat $LOG_FILE"