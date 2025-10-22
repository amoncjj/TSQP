import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = "Hello, world!"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_LENGTH = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_RESULT_PATH = "tee_only_results.json"
DEFAULT_MODEL_PATH = "weights/llama3.2-1b"

PROMPT_LIST_ENV = "LLAMA_PROMPT_LIST"
PROMPT_PATH_ENV = "LLAMA_PROMPT_PATH"
BATCH_SIZE_ENV = "LLAMA_TEE_BATCH_SIZE"
MAX_LENGTH_ENV = "LLAMA_MAX_LENGTH"
TEMPERATURE_ENV = "LLAMA_TEMPERATURE"
TOP_P_ENV = "LLAMA_TOP_P"
MODEL_PATH_ENV = "LLAMA_MODEL_PATH"
RESULT_PATH_ENV = "LLAMA_TEE_RESULT_PATH"


def read_prompts() -> List[str]:
    prompt_list_env = os.environ.get(PROMPT_LIST_ENV)
    prompt_path_env = os.environ.get(PROMPT_PATH_ENV)

    if prompt_list_env:
        try:
            return json.loads(prompt_list_env)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse LLAMA_PROMPT_LIST: {exc}")

    if prompt_path_env and os.path.exists(prompt_path_env):
        with open(prompt_path_env, "r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
            if prompts:
                return prompts

    return [DEFAULT_PROMPT]


def resolve_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {exc}")


def resolve_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for {name}: {exc}")


def load_model_and_tokenizer(model_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(torch.device("cpu"))
    model.eval()

    return model, tokenizer


def run_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_length: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Iterable[str]]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {
        "prompts": prompts,
        "completions": decoded,
    }


def benchmark(prompts: List[str], model_path: str, output_path: str) -> Dict[str, float]:
    max_length = resolve_int_env(MAX_LENGTH_ENV, DEFAULT_MAX_LENGTH)
    temperature = resolve_float_env(TEMPERATURE_ENV, DEFAULT_TEMPERATURE)
    top_p = resolve_float_env(TOP_P_ENV, DEFAULT_TOP_P)

    model, tokenizer = load_model_and_tokenizer(model_path)

    start = time.perf_counter()
    generation_payload = run_generation(model, tokenizer, prompts, max_length, temperature, top_p)
    elapsed = time.perf_counter() - start

    results = {
        "mode": "tee-only",
        "prompt_count": len(prompts),
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "time_seconds": elapsed,
        "outputs": generation_payload["completions"],
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    prompts = read_prompts()
    batch_size = resolve_int_env(BATCH_SIZE_ENV, DEFAULT_BATCH_SIZE)
    prompts = (prompts * ((batch_size + len(prompts) - 1) // len(prompts)))[:batch_size]

    model_path = os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
    output_path = os.environ.get(RESULT_PATH_ENV, DEFAULT_RESULT_PATH)

    result = benchmark(prompts, model_path, output_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
