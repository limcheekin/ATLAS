@echo off
setlocal

:: --- LLM Inference (CPU only) ---

:: Set defaults
if not defined ATLAS_MODELS_DIR set ATLAS_MODELS_DIR=C:\dev\ws\docker\llama.cpp\models
if not defined ATLAS_MODEL_FILE set ATLAS_MODEL_FILE=Marco-Mini-Instruct.Q4_K_M.gguf
if not defined ATLAS_CTX_SIZE set ATLAS_CTX_SIZE=32768
if not defined ATLAS_LLAMA_PORT set ATLAS_LLAMA_PORT=8080

llama-server ^
  --model "%ATLAS_MODELS_DIR%\%ATLAS_MODEL_FILE%" ^
  --host 0.0.0.0 --port %ATLAS_LLAMA_PORT% ^
  --ctx-size %ATLAS_CTX_SIZE% ^
  --n-gpu-layers 0 ^
  --no-mmap

endlocal
