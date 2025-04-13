#!/bin/bash

# Configuration variables
VENV_PATH="/home/jay/MLOps/vLLM_host/.vllm"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HOST="0.0.0.0"
PORT="8000"
MAX_MODEL_LEN="512"
QUANTIZATION="bitsandbytes"
DTYPE="auto"
GPU_MEMORY_UTILIZATION="0.5"
LOG_FILE="/home/jay/MLOps/vLLM_host/vllm_server.log"
CUDA_VISIBLE_DEVICES="1"  # Use GPU 1, adjust if needed

# Function to check if a process is already running on the port
check_port() {
    if netstat -tulnp 2>/dev/null | grep -q ":${PORT} "; then
        echo "Error: Port ${PORT} is already in use. Please stop the existing process or use a different port."
        exit 1
    fi
}

# Check if nvidia-smi is available and GPU is accessible
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. Assuming CPU mode or checking GPU access."
    else
        if ! nvidia-smi | grep -q "NVIDIA"; then
            echo "Error: No NVIDIA GPU detected. Remove CUDA_VISIBLE_DEVICES or check GPU setup."
            exit 1
        fi
    fi
}

# Activate virtual environment
activate_venv() {
    if [ -d "$VENV_PATH" ]; then
        source "${VENV_PATH}/bin/activate"
        echo "Activated virtual environment at ${VENV_PATH}"
    else
        echo "Error: Virtual environment not found at ${VENV_PATH}"
        exit 1
    fi
}

# Check if vLLM is installed
check_vllm() {
    if ! pip show vllm &> /dev/null; then
        echo "Error: vLLM not installed in the virtual environment."
        echo "Please run 'pip install vllm' in ${VENV_PATH}"
        exit 1
    fi
}

# Start vLLM server
start_server() {
    echo "Starting vLLM server with model ${MODEL} on ${HOST}:${PORT}..."
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --quantization "${QUANTIZATION}" \
        --dtype "${DTYPE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        > "${LOG_FILE}" 2>&1 &
    local pid=$!
    sleep 5  # Wait for server to start
    if ps -p $pid > /dev/null; then
        echo "vLLM server started successfully (PID: ${pid}). Logs at ${LOG_FILE}"
    else
        echo "Error: vLLM server failed to start. Check ${LOG_FILE} for details."
        cat "${LOG_FILE}"
        exit 1
    fi
}

# Main execution
echo "Initializing vLLM server setup..."

check_port
check_gpu
activate_venv
check_vllm
start_server

echo "Setup complete. You can test the server with:"
echo "curl -X POST http://${HOST}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello, world!\", \"max_tokens\": 50}'"