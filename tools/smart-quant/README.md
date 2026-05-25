# smart-quant

AI-assisted quantization advisor for llama.cpp. Detects your system's RAM and VRAM, then queries an OpenAI-compatible AI provider to recommend the best quantization strategy for fitting a model on your hardware.

## Features

- Auto-detects system RAM and NVIDIA GPU VRAM
- Works with any OpenAI-compatible API (OpenAI, local llama-server, Together, Groq, etc.)
- API key is optional (for local/unauthenticated endpoints)
- Recommends quantization type, GPU layer count, and context length
- Provides a ready-to-use llama-cli/llama-server command
- Suggests alternative quantizations with trade-off descriptions
- JSON output mode for scripting

## Requirements

- Python 3.10+
- No additional Python packages required (uses only the standard library)
- `nvidia-smi` for GPU detection (optional, falls back to CPU-only)

## Usage

```bash
# Using OpenAI (requires API key)
python tools/smart-quant/smart_quant.py \
    --model meta-llama/Llama-3.1-8B \
    --api-key sk-... \
    --ai-model gpt-4o-mini

# Using a local llama-server as the AI provider (no key needed)
python tools/smart-quant/smart_quant.py \
    --model mistralai/Mistral-7B-v0.1 \
    --api-url http://localhost:8080/v1 \
    --ai-model local-model

# Using any OpenAI-compatible provider (e.g. Together AI)
python tools/smart-quant/smart_quant.py \
    --model Qwen/Qwen2-72B \
    --api-url https://api.together.xyz/v1 \
    --api-key $TOGETHER_API_KEY \
    --ai-model meta-llama/Meta-Llama-3-70B

# Override auto-detected hardware specs
python tools/smart-quant/smart_quant.py \
    --model meta-llama/Llama-3.1-70B \
    --ram 32768 --vram 24576 \
    --api-key sk-...

# JSON output for scripting
python tools/smart-quant/smart_quant.py \
    --model Llama-3.1-8B \
    --api-key sk-... --json
```

## Options

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Model name or HuggingFace ID (required) |
| `--params` | Model parameter count in billions (auto-detected from name) |
| `--context-length`, `-c` | Desired context length (default: 4096) |
| `--api-url` | OpenAI-compatible API base URL (default: `https://api.openai.com/v1`) |
| `--api-key` | API key (also reads `SMART_QUANT_API_KEY` env var) |
| `--ai-model` | AI model for recommendations (default: `gpt-4o-mini`) |
| `--ram` | Override detected RAM in MiB |
| `--vram` | Override detected VRAM in MiB |
| `--json` | Output raw JSON for scripting |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SMART_QUANT_API_KEY` | API key (alternative to `--api-key`) |

## How It Works

1. **System detection**: Reads `/proc/meminfo` (Linux), `sysctl` (macOS), or `wmic` (Windows) for RAM. Runs `nvidia-smi` for NVIDIA GPU VRAM.
2. **Model analysis**: Parses the model name to extract parameter count and estimates memory footprint at each quantization level.
3. **AI consultation**: Sends system specs, model info, and all available quantization types to the AI provider with a structured prompt.
4. **Recommendation**: The AI returns a JSON response with the optimal quantization type, estimated size, GPU offloading strategy, and a ready-to-use command.

## Example Output

```
============================================================
  SMART-QUANT RECOMMENDATION
============================================================

  Quantization type:  Q4_K_M
  Estimated size:     4.58 GiB
  GPU layers:         all
  Context length:     4096
  Fits in VRAM:       True
  Hybrid CPU+GPU:     False

  Reasoning:
    With 24 GiB of VRAM on your RTX 4090, the
    Llama-3.1-8B model quantized to Q4_K_M (4.58
    GiB) fits comfortably with room for KV cache at
    4096 context. Q4_K_M offers the best balance of
    quality and size for your hardware.

  Suggested command:
    llama-cli -m model-Q4_K_M.gguf -ngl 99 -c 4096

  Alternatives:
    - Q5_K_M: Slightly better quality (+0.75 GiB)
    - Q6_K: High quality but uses 6.14 GiB

============================================================
```
