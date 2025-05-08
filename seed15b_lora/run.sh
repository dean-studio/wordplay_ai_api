#!/bin/bash
# 파일명: run_training_with_tensorboard.sh

# 로그 디렉토리 설정
TENSORBOARD_LOG_DIR="./tensorboard_logs"
TRAINING_LOG_DIR="./logs"
mkdir -p $TENSORBOARD_LOG_DIR
mkdir -p $TRAINING_LOG_DIR

# 현재 시간을 파일명으로 사용
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRAINING_LOG="$TRAINING_LOG_DIR/training_$TIMESTAMP.log"
TENSORBOARD_LOG="$TRAINING_LOG_DIR/tensorboard_$TIMESTAMP.log"

echo "===================================================="
echo "CLOVA X 1.5B LoRA 학습 및 TensorBoard 모니터링 시작"
echo "===================================================="
echo "시작 시간: $(date)"
echo "로그 파일: $TRAINING_LOG"
echo "TensorBoard 로그 디렉토리: $TENSORBOARD_LOG_DIR"

# TensorBoard 실행 (포트 5000 사용)
echo "TensorBoard 시작 중..."
tensorboard --logdir=$TENSORBOARD_LOG_DIR --host=0.0.0.0 --port=5000 > $TENSORBOARD_LOG 2>&1 &
TENSORBOARD_PID=$!

# TensorBoard가 시작되기를 잠시 기다림
sleep 3

# TensorBoard 실행 확인
if ps -p $TENSORBOARD_PID > /dev/null; then
    echo "TensorBoard가 성공적으로 시작되었습니다 (PID: $TENSORBOARD_PID)"
    echo "접속 URL: http://$(hostname -I | awk '{print $1}'):5000"
else
    echo "TensorBoard 시작에 실패했습니다. 로그를 확인하세요: $TENSORBOARD_LOG"
    exit 1
fi

# 학습 스크립트 실행
echo "학습 시작 중..."
python lora.py > $TRAINING_LOG 2>&1

# 학습 종료 상태 확인
TRAINING_EXIT_CODE=$?
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "학습이 성공적으로 완료되었습니다!"
else
    echo "학습이 오류와 함께 종료되었습니다 (종료 코드: $TRAINING_EXIT_CODE)"
    echo "로그 파일을 확인하세요: $TRAINING_LOG"
fi

# TensorBoard 상태 출력
echo "TensorBoard는 계속 실행 중입니다 (PID: $TENSORBOARD_PID)"
echo "종료하려면 다음 명령어를 사용하세요: kill $TENSORBOARD_PID"

echo "===================================================="
echo "실행 요약:"
echo "- 시작 시간: $(date -d @$(($(date +%s) - $SECONDS)))"
echo "- 종료 시간: $(date)"
echo "- 총 소요 시간: $((SECONDS/3600))시간 $(((SECONDS%3600)/60))분 $((SECONDS%60))초"
echo "- 학습 로그: $TRAINING_LOG"
echo "- TensorBoard 로그: $TENSORBOARD_LOG"
echo "- TensorBoard PID: $TENSORBOARD_PID"
echo "===================================================="
echo "로그 확인 명령어:"
echo "tail -f $TRAINING_LOG"
echo "===================================================="