import torch
def main():
    print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); b = a @ a; torch.cuda.synchronize(); print("bf16 matmul ok")
    import flash_attn
    from flash_attn import flash_attn_func
    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    o = flash_attn_func(q, q, q, causal=True); o.sum().backward(); torch.cuda.synchronize(); print("flash_attn", flash_attn.__version__, "fwd+bwd ok")
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen3-0.6B", gpu_memory_utilization=0.2, max_model_len=1024, dtype="bfloat16")
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
    print("vllm gen ok:", repr(out[0].outputs[0].text))
if __name__ == "__main__":
    main()
