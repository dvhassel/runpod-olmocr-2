import runpod
import base64, io
from PIL import Image
from vllm import LLM, SamplingParams
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

MODEL_NAME = "allenai/olmOCR-2-7B-1025-FP8"

_llm = None
_llm_error = None

def get_llm():
    global _llm, _llm_error
    if _llm is not None:
        return _llm
    if _llm_error is not None:
        # fail fast on subsequent jobs instead of repeatedly trying
        raise RuntimeError(_llm_error)

    try:
        _llm = LLM(
            model=MODEL_NAME,
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

    job_input = job["input"]
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided in 'image' field."}

    # Decode and Resize
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = resize_image_safely(image, max_edge=1288)
    
    # Convert resized image back to base64 for the chat template
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    resized_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Build the exact prompt expected by the FP8 model
    prompt = build_no_anchoring_v4_yaml_prompt()
        
    # Construct OpenAI-style messages (The standard way Qwen2.5-VL handles images in vLLM)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{resized_b64}"}}
            ]
        }
    ]
        
    sampling_params = SamplingParams(
        temperature=job_input.get("temperature", 0.1),
        max_tokens=job_input.get("max_tokens", 2048),
    )
    outputs = llm.chat(messages, sampling_params=sampling_params)
    return {"text": outputs[0].outputs[0].text}
    
runpod.serverless.start({'handler': handler })
