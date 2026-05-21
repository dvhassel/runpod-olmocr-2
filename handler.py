import runpod
import base64
import io
from PIL import Image
from vllm import LLM, SamplingParams
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# Correct model name as specified in the allenai/olmocr repository
MODEL_NAME = "allenai/olmOCR-2-7B-1025-FP8"

# Model Initialization using the correct fp8 quantization
llm = LLM(
    model=MODEL_NAME, 
    #quantization="fp8", 
    max_model_len=4096, 
    gpu_memory_utilization=0.85,
    trust_remote_code=True
)

def resize_image_safely(image, max_edge=1288):
    """Resizes the image if it exceeds the maximum edge length, maintaining aspect ratio."""
    width, height = image.size
    if max(width, height) > max_edge:
        scaling_factor = max_edge / float(max(width, height))
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def handler(job):
    job_input = job['input']
    image_b64 = job_input.get("image")
    
    if not image_b64:
        return {"error": "No image provided in 'image' field."}

    try:
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
            max_tokens=job_input.get("max_tokens", 2048)
        )
        
        # Generate the output using the chat interface
        outputs = llm.chat(messages, sampling_params=sampling_params)
        return {"text": outputs[0].outputs[0].text}

    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

runpod.serverless.start({'handler': handler })
