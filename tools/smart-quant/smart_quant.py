#!/usr/bin/env python3
"""smart-quant: AI-assisted quantization advisor for llama.cpp.

Detects system RAM/VRAM and queries an OpenAI-compatible AI provider
to recommend the best quantization strategy for fitting a model on
the user's hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger("smart-quant")

QUANTIZATION_TYPES: list[dict[str, object]] = [
    {"name": "IQ1_S",   "bpw": 2.00,  "desc": "1-bit (extreme). Very lossy."},
    {"name": "IQ1_M",   "bpw": 2.15,  "desc": "1-bit (moderate). Very lossy."},
    {"name": "IQ2_XXS", "bpw": 2.38,  "desc": "2-bit (extra extra small). Very lossy."},
    {"name": "IQ2_XS",  "bpw": 2.59,  "desc": "2-bit (extra small). Very lossy."},
    {"name": "IQ2_S",   "bpw": 2.74,  "desc": "2-bit (small). Very lossy."},
    {"name": "IQ2_M",   "bpw": 2.93,  "desc": "2-bit (medium). Lossy."},
    {"name": "Q2_K_S",  "bpw": 2.97,  "desc": "2-bit k-quant (small). Lossy."},
    {"name": "Q2_K",    "bpw": 3.16,  "desc": "2-bit k-quant. Lossy."},
    {"name": "IQ3_XXS", "bpw": 3.25,  "desc": "3-bit (extra extra small). Moderate loss."},
    {"name": "IQ3_XS",  "bpw": 3.50,  "desc": "3-bit (extra small). Moderate loss."},
    {"name": "Q3_K_S",  "bpw": 3.64,  "desc": "3-bit k-quant (small). Moderate loss."},
    {"name": "IQ3_S",   "bpw": 3.66,  "desc": "3-bit (small). Moderate loss."},
    {"name": "IQ3_M",   "bpw": 3.76,  "desc": "3-bit (medium). Moderate loss."},
    {"name": "Q3_K_M",  "bpw": 4.00,  "desc": "3-bit k-quant (medium). Balanced."},
    {"name": "Q3_K_L",  "bpw": 4.30,  "desc": "3-bit k-quant (large). Balanced."},
    {"name": "IQ4_XS",  "bpw": 4.46,  "desc": "4-bit (extra small). Good quality."},
    {"name": "Q4_K_S",  "bpw": 4.67,  "desc": "4-bit k-quant (small). Good quality."},
    {"name": "IQ4_NL",  "bpw": 4.68,  "desc": "4-bit (non-linear). Good quality."},
    {"name": "Q4_K_M",  "bpw": 4.89,  "desc": "4-bit k-quant (medium). Recommended."},
    {"name": "Q5_K_S",  "bpw": 5.57,  "desc": "5-bit k-quant (small). High quality."},
    {"name": "Q5_K_M",  "bpw": 5.70,  "desc": "5-bit k-quant (medium). High quality."},
    {"name": "Q6_K",    "bpw": 6.56,  "desc": "6-bit k-quant. Very high quality."},
    {"name": "Q8_0",    "bpw": 8.50,  "desc": "8-bit. Near-lossless."},
    {"name": "F16",     "bpw": 16.00, "desc": "16-bit float. Lossless."},
    {"name": "BF16",    "bpw": 16.00, "desc": "Brain float 16. Lossless."},
    {"name": "F32",     "bpw": 32.00, "desc": "32-bit float. Full precision."},
]


@dataclass
class SystemInfo:
    ram_mb: int = 0
    gpu_devices: list[dict[str, object]] = field(default_factory=list)
    total_vram_mb: int = 0
    os_name: str = ""
    cpu: str = ""

    def summary(self) -> str:
        lines = [
            f"OS: {self.os_name}",
            f"CPU: {self.cpu}",
            f"System RAM: {self.ram_mb} MiB"
            f" ({self.ram_mb / 1024:.1f} GiB)",
        ]
        if self.gpu_devices:
            for gpu in self.gpu_devices:
                lines.append(
                    f"GPU: {gpu['name']}"
                    f" - VRAM: {gpu['vram_mb']} MiB"
                    f" ({int(gpu['vram_mb']) / 1024:.1f} GiB)"  # type: ignore[arg-type]
                )
            lines.append(
                f"Total VRAM: {self.total_vram_mb} MiB"
                f" ({self.total_vram_mb / 1024:.1f} GiB)"
            )
        else:
            lines.append("GPU: None detected (CPU-only mode)")
        return "\n".join(lines)


@dataclass
class ModelInfo:
    name: str = ""
    params_billion: float = 0.0
    context_length: int = 0
    architecture: str = ""

    def summary(self) -> str:
        parts = [f"Model: {self.name}"]
        if self.params_billion > 0:
            parts.append(f"Parameters: {self.params_billion:.1f}B")
        if self.context_length > 0:
            parts.append(f"Context length: {self.context_length}")
        if self.architecture:
            parts.append(f"Architecture: {self.architecture}")
        return "\n".join(parts)


def detect_system_info() -> SystemInfo:
    """Detect system RAM and GPU VRAM."""
    info = SystemInfo()
    info.os_name = f"{platform.system()} {platform.release()}"
    info.cpu = platform.processor() or platform.machine()

    # Detect RAM
    info.ram_mb = _detect_ram_mb()

    # Detect GPU VRAM
    info.gpu_devices = _detect_gpu_devices()
    info.total_vram_mb = sum(
        int(g["vram_mb"]) for g in info.gpu_devices  # type: ignore[arg-type]
    )

    return info


def _detect_ram_mb() -> int:
    """Detect total system RAM in MiB."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except OSError:
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True,
            )
            return int(result.stdout.strip()) // (1024 * 1024)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass
    elif system == "Windows":
        try:
            result = subprocess.run(
                [
                    "wmic", "ComputerSystem", "get",
                    "TotalPhysicalMemory", "/value",
                ],
                capture_output=True, text=True, check=True,
            )
            for line in result.stdout.splitlines():
                if "TotalPhysicalMemory" in line:
                    val = line.split("=")[1].strip()
                    return int(val) // (1024 * 1024)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass
    return 0


def _detect_gpu_devices() -> list[dict[str, object]]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    devices: list[dict[str, object]] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return devices
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                devices.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_mb": int(parts[2]),
                })
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return devices


def parse_model_identifier(model_str: str) -> ModelInfo:
    """Extract model info from a model name/path string.

    Attempts to parse parameter count and other details from common
    naming conventions (e.g. 'meta-llama/Llama-3.1-8B',
    'Qwen/Qwen2-72B-Instruct', 'mistral-7b-v0.1').
    """
    info = ModelInfo(name=model_str)

    param_match = re.search(
        r'(\d+(?:\.\d+)?)\s*[bB](?:illion)?(?:\b|[-_])',
        model_str,
    )
    if param_match:
        info.params_billion = float(param_match.group(1))

    return info


def estimate_model_size_mb(params_billion: float, bpw: float) -> float:
    """Estimate GGUF model file size in MiB given params and bits-per-weight."""
    total_bits = params_billion * 1e9 * bpw
    total_bytes = total_bits / 8
    return total_bytes / (1024 * 1024)


def build_ai_prompt(
    system_info: SystemInfo,
    model_info: ModelInfo,
    context_length: int,
) -> str:
    """Build the prompt to send to the AI provider."""
    quant_table = "\n".join(
        f"  - {q['name']}: {q['bpw']} bits/weight - {q['desc']}"
        for q in QUANTIZATION_TYPES
    )

    size_estimates = ""
    if model_info.params_billion > 0:
        size_estimates = "\nEstimated model sizes at various quantizations:\n"
        for q in QUANTIZATION_TYPES:
            size_mb = estimate_model_size_mb(
                model_info.params_billion, float(q["bpw"]),
            )
            size_estimates += (
                f"  - {q['name']}: {size_mb:.0f} MiB"
                f" ({size_mb / 1024:.1f} GiB)\n"
            )

    return f"""You are an expert on llama.cpp model quantization and deployment.

The user wants to run a model on their system. Based on the system specs and \
model info below, recommend the best quantization strategy.

=== SYSTEM SPECS ===
{system_info.summary()}

=== MODEL INFO ===
{model_info.summary()}
Desired context length: {context_length}
{size_estimates}
=== AVAILABLE QUANTIZATION TYPES ===
{quant_table}

=== IMPORTANT CONSIDERATIONS ===
- The model weights must fit in available memory (VRAM + RAM for hybrid mode)
- GPU offloading is preferred for speed: put as many layers on GPU as possible
- KV cache memory usage scales with context length (~2 bytes per element)
- KV cache size (MiB) ~ (n_layers * 2 * n_heads * head_dim * context \
* 2) / 1048576
- Leave ~500-1000 MiB headroom for the runtime and KV cache on GPU
- Higher quantization = better quality but larger size
- For GPU-only: model + KV cache must fit in VRAM
- For hybrid CPU+GPU: some layers can overflow to RAM but this is slower
- Q4_K_M is generally the sweet spot for quality vs. size
- For very constrained systems, IQ quantizations offer better quality at \
low bit rates

=== INSTRUCTIONS ===
Respond with a JSON object (no markdown, no code fences) with these fields:
{{
  "recommended_quant": "<quantization type name>",
  "estimated_size_gib": <number>,
  "gpu_layers": <number or "all">,
  "context_length": <recommended context length>,
  "will_fit_in_vram": <true/false>,
  "hybrid_mode": <true/false>,
  "reasoning": "<brief explanation of your recommendation>",
  "llama_cpp_command": "<example llama-cli or llama-server command>",
  "alternative_quants": [
    {{
      "name": "<type>",
      "trade_off": "<brief description>"
    }}
  ]
}}
"""


def query_ai_provider(
    prompt: str,
    api_url: str,
    api_key: str | None,
    ai_model: str,
) -> dict[str, object]:
    """Send prompt to an OpenAI-compatible chat completions endpoint."""
    endpoint = api_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"

    payload = {
        "model": ai_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a llama.cpp quantization expert. "
                    "Respond only with valid JSON, "
                    "no markdown formatting."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data, headers=headers, method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error("API returned HTTP %d: %s", e.code, body)
        sys.exit(1)
    except urllib.error.URLError as e:
        logger.error("Could not connect to API: %s", e.reason)
        sys.exit(1)

    content = resp_data["choices"][0]["message"]["content"]

    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    return json.loads(content)  # type: ignore[no-any-return]


def format_recommendation(rec: dict[str, object]) -> str:
    """Format the AI recommendation for display."""
    lines = [
        "",
        "=" * 60,
        "  SMART-QUANT RECOMMENDATION",
        "=" * 60,
        "",
        f"  Quantization type:  {rec.get('recommended_quant', 'N/A')}",
        f"  Estimated size:     {rec.get('estimated_size_gib', 'N/A')} GiB",
        f"  GPU layers:         {rec.get('gpu_layers', 'N/A')}",
        f"  Context length:     {rec.get('context_length', 'N/A')}",
        f"  Fits in VRAM:       {rec.get('will_fit_in_vram', 'N/A')}",
        f"  Hybrid CPU+GPU:     {rec.get('hybrid_mode', 'N/A')}",
        "",
        "  Reasoning:",
    ]

    reasoning = str(rec.get("reasoning", ""))
    for i in range(0, len(reasoning), 56):
        lines.append(f"    {reasoning[i:i + 56]}")

    cmd = rec.get("llama_cpp_command", "")
    if cmd:
        lines.extend(["", "  Suggested command:", f"    {cmd}"])

    alternatives = rec.get("alternative_quants", [])
    if alternatives:
        lines.extend(["", "  Alternatives:"])
        for alt in alternatives:  # type: ignore[union-attr]
            lines.append(
                f"    - {alt['name']}: {alt['trade_off']}"  # type: ignore[index]
            )

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "smart-quant: AI-assisted quantization advisor for llama.cpp. "
            "Detects your system's RAM/VRAM and queries an AI provider to "
            "recommend the best quantization for fitting a model on your "
            "hardware."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Use OpenAI
  %(prog)s --model meta-llama/Llama-3.1-8B \\
      --api-key sk-... --ai-model gpt-4o-mini

  # Use a local llama-server as the AI provider (no key needed)
  %(prog)s --model mistralai/Mistral-7B-v0.1 \\
      --api-url http://localhost:8080/v1 --ai-model local-model

  # Use any OpenAI-compatible provider
  %(prog)s --model Qwen/Qwen2-72B \\
      --api-url https://api.together.xyz/v1 \\
      --api-key $TOGETHER_API_KEY --ai-model meta-llama/Meta-Llama-3-70B

  # Override auto-detected specs
  %(prog)s --model meta-llama/Llama-3.1-70B \\
      --ram 32768 --vram 24576 --api-key sk-...

  # Output as JSON for scripting
  %(prog)s --model Llama-3.1-8B --api-key sk-... --json
""",
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help=(
            "Model name or HuggingFace identifier "
            "(e.g. 'meta-llama/Llama-3.1-8B', 'mistral-7b')"
        ),
    )
    parser.add_argument(
        "--params",
        type=float,
        default=None,
        help=(
            "Model parameter count in billions "
            "(auto-detected from model name if possible)"
        ),
    )
    parser.add_argument(
        "--context-length", "-c",
        type=int,
        default=4096,
        help="Desired context length (default: 4096)",
    )
    parser.add_argument(
        "--api-url",
        default="https://api.openai.com/v1",
        help=(
            "OpenAI-compatible API base URL "
            "(default: https://api.openai.com/v1)"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for the AI provider "
            "(optional, also reads SMART_QUANT_API_KEY env var)"
        ),
    )
    parser.add_argument(
        "--ai-model",
        default="gpt-4o-mini",
        help="AI model to use for recommendations (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--ram",
        type=int,
        default=None,
        help="Override auto-detected RAM in MiB",
    )
    parser.add_argument(
        "--vram",
        type=int,
        default=None,
        help="Override auto-detected total VRAM in MiB",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON recommendation (for scripting)",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s",
    )
    args = parse_args()

    # Detect system info
    logger.info("Detecting system hardware...")
    sys_info = detect_system_info()

    if args.ram is not None:
        sys_info.ram_mb = args.ram
    if args.vram is not None:
        sys_info.total_vram_mb = args.vram
        if not sys_info.gpu_devices:
            sys_info.gpu_devices = [
                {"index": 0, "name": "User-specified GPU", "vram_mb": args.vram},
            ]

    for line in sys_info.summary().splitlines():
        logger.info("%s", line)

    # Parse model info
    model_info = parse_model_identifier(args.model)
    if args.params is not None:
        model_info.params_billion = args.params

    if model_info.params_billion == 0:
        logger.warning(
            "Could not detect parameter count from model name. "
            "Use --params to specify (e.g. --params 7).",
        )

    for line in model_info.summary().splitlines():
        logger.info("%s", line)

    # Resolve API key
    api_key = args.api_key or os.environ.get("SMART_QUANT_API_KEY", "")

    # Build prompt and query AI
    prompt = build_ai_prompt(sys_info, model_info, args.context_length)

    logger.info("Querying AI advisor (%s)...", args.ai_model)
    recommendation = query_ai_provider(
        prompt, args.api_url, api_key, args.ai_model,
    )

    # Output
    if args.json_output:
        sys.stdout.write(json.dumps(recommendation, indent=2) + "\n")
    else:
        sys.stdout.write(format_recommendation(recommendation) + "\n")


if __name__ == "__main__":
    main()
