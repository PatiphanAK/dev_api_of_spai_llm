import os
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="SuperAI Demo")

MODEL_DIR = os.environ.get("MODEL_DIR", "/content/model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_model():
    """Load tokenizer + model from safetensors weights in MODEL_DIR (cached)."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        use_safetensors=True,  # force loading weights from .safetensors
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


class InferRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    do_sample: bool = True


class InferResponse(BaseModel):
    output: str
    prompt_tokens: int
    completion_tokens: int


@app.get("/")
def read_root():
    return {"message": "Hello from SuperAI FastAPI demo"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    try:
        tokenizer, model = load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}")

    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Strip the prompt tokens so we only decode the newly generated text.
    new_tokens = generated[0][prompt_tokens:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return InferResponse(
        output=output,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(new_tokens.shape[-1]),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
