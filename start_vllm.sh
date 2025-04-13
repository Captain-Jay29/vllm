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
LOG_DIR="/home/jay/MLOps/vLLM_host/logs"
LOG_FILE="${LOG_DIR}/vllm_server.log"
CUDA_VISIBLE_DEVICES="1"
PROMETHEUS_PATH="/home/jay/MLOps/vLLM_host/prometheus"
NODE_EXPORTER_PATH="/home/jay/MLOps/vLLM_host/node_exporter"
DCGM_EXPORTER_LOG="/home/jay/MLOps/vLLM_host/dcgm_exporter.log"

# Create log directory
mkdir -p "${LOG_DIR}"

# Function to check if a process is already running on a port
check_port() {
    if netstat -tulnp 2>/dev/null | grep -q ":${1} "; then
        echo "Error: Port ${1} is already in use."
        exit 1
    fi
}

# Check GPU
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found."
    else
        if ! nvidia-smi | grep -q "NVIDIA"; then
            echo "Error: No NVIDIA GPU detected."
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

# Check vLLM
check_vllm() {
    if ! pip show vllm &> /dev/null; then
        echo "Error: vLLM not installed."
        exit 1
    fi
}

# Start monitoring tools
start_monitoring() {
    echo "Starting monitoring tools..."
    # Prometheus
    check_port 9090
    nohup "${PROMETHEUS_PATH}/prometheus" \
        --config.file="${PROMETHEUS_PATH}/prometheus.yml" \
        > "${LOG_DIR}/prometheus.log" 2>&1 &
    echo "Prometheus started (PID: $!)."

    # Node Exporter
    check_port 9100
    nohup "${NODE_EXPORTER_PATH}/node_exporter" \
        > "${LOG_DIR}/node_exporter.log" 2>&1 &
    echo "Node Exporter started (PID: $!)."

    # DCGM Exporter
    check_port 9400
    nohup dcgm-exporter > "${DCGM_EXPORTER_LOG}" 2>&1 &
    echo "DCGM Exporter started (PID: $!)."
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
    sleep 5
    if ps -p $pid > /dev/null; then
        echo "vLLM server started (PID: ${pid}). Logs at ${LOG_FILE}"
    else
        echo "Error: vLLM server failed to start."
        cat "${LOG_FILE}"
        exit 1
    fi
}

# Main execution
echo "Initializing vLLM server setup..."
check_port 8000
check_gpu
activate_venv
check_vllm
start_monitoring
start_server

echo "Setup complete. Test with:"
echo "curl -X POST http://${HOST}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello, world!\", \"max_tokens\": 50}'"
echo "Monitor at: http://localhost:3000 (Grafana), http://localhost:9090 (Prometheus)"