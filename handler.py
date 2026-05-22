# handler.py
import os
import io
import base64

import runpod
from PIL import Image
from vllm import LLM, SamplingParams

from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# Runpod cached HF models live here on Serverless workers
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
MODEL_ID = "allenai/olmocr-2-7b-1025"  # use the canonical repo name (case-insensitive on HF, but cache path expects this form)

# Force offline mode (prevents accidental network calls)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_llm = None
_llm_error = None


def resolve_snapshot_path(model_id: str) -> str:
    """
    Resolve the local snapshot path for a Runpod-cached Hugging Face model.
    Looks in /runpod-volume/huggingface-cache/hub following HF cache conventions.
    """
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    # Prefer the snapshot referenced by refs/main
    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    # Fall back to the first snapshot directory present
    if os.path.isdir(snapshots_dir):
        versions = sorted(
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        )
        if versions:
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(
        f"Cached model not found on disk for '{model_id}'. "
        f"Expected under: {model_root}. "
        f"Make sure your endpoint has Model Caching enabled for this model."
    )


def resize_image_safely(image: Image.Image, max_edge: int = 1288) -> Image.Image:
    """
    Downscale an image so its longest edge is <= max_edge, preserving aspect ratio.
    """
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), resample=Image.BICUBIC)


def get_llm() -> LLM:
    global _llm, _llm_error
    if _llm is not None:
        return _llm
    if _llm_error is not None:
        raise RuntimeError(_llm_error)

    try:
        local_model_path = resolve_snapshot_path(MODEL_ID)
        print(f"[init] Loading model from local cache: {local_model_path}", flush=True)

        _llm = LLM(
            model=local_model_path,        # IMPORTANT: load from local cached snapshot path
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        return _llm
    except Exception as e:
        _llm_error = f"LLM init failed: {e}"
        raise


def handler(job):
    llm = get_llm()

    job_input = job.get("input", {}) or {}
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided in 'image' field."}

    # Decode and resize
    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image data: {e}"}

    image = resize_image_safely(image, max_edge=1288)

    # Convert resized image back to base64 for the chat template
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    resized_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Build the prompt expected by olmOCR
    prompt = build_no_anchoring_v4_yaml_prompt()

    # OpenAI-style messages with a data: URL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{resized_b64}"}},
            ],
        }
    ]

    sampling_params = SamplingParams(
        temperature=float(job_input.get("temperature", 0.1)),
        max_tokens=int(job_input.get("max_tokens", 2048)),
    )

    outputs = llm.chat(messages, sampling_params=sampling_params)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

    return {"text": text}


runpod.serverless.start({"handler": handler})
