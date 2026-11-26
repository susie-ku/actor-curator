"""Forward-pass check on sm_120: vLLM prompt logprobs vs HF (sdpa, bf16) for the same prompt. vLLM first, then HF."""
import os, math, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    q = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
         "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
         "User: Let $p(x)$ be a quadratic polynomial with integer coefficients which has $4-\\sqrt{11}$ as a root. "
         "Compute $\\frac{p(3)}{p(4)}.$ Put your final answer within \\boxed{}. \nAssistant: Let me solve this step by step. "
         "Since $p(x)$ has integer coefficients and $4-\\sqrt{11}$ is a root, its conjugate $4+\\sqrt{11}$ must also be a root.\n\n"
         "So $p(x) = a(x^2 - 8x + 5)$ for some integer $a$. Then $p(3) = a(9 - 24 + 5) = -10a$ and $p(4) = a(16 - 32 + 5) = -11a$.")
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids
    llm = LLM(model="Qwen/Qwen2.5-3B", gpu_memory_utilization=0.3, max_model_len=2048, dtype="bfloat16", seed=1, enable_prefix_caching=False)
    o = llm.generate([prompt], SamplingParams(max_tokens=1, prompt_logprobs=0, logprobs=20, temperature=0))[0]
    vl = {}
    for t in range(1, ids.shape[1]):
        d = o.prompt_logprobs[t]
        if d:
            tid = ids[0, t].item()
            if tid in d: vl[t] = d[tid].logprob
    last = {k: v.logprob for k, v in o.outputs[0].logprobs[0].items()}
    del llm; import gc; gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    with torch.no_grad():
        hf_logp = torch.log_softmax(model(ids.cuda()).logits[0].float(), -1).cpu()
    diffs = [abs(vl[t] - hf_logp[t - 1, ids[0, t].item()].item()) for t in vl]
    print(f"prompt tokens compared: {len(diffs)}  mean |Δlogp|={sum(diffs)/len(diffs):.4f}  max={max(diffs):.4f}  (bf16 noise is ~0.01-0.05)")
    hf_top = torch.topk(hf_logp[-1], 5)
    print("HF   top5 next:", [(tok.decode([i]), round(v.item(), 3)) for v, i in zip(hf_top.values, hf_top.indices)])
    print("vLLM top5 next:", sorted([(tok.decode([k]), round(v, 3)) for k, v in last.items()], key=lambda x: -x[1])[:5])
    kl = sum(math.exp(v) * (v - hf_logp[-1, k].item()) for k, v in last.items())
    print(f"KL(vLLM||HF) over vLLM top-20 next tokens: {kl:.5f}")
    # also an eager-attention HF reference to bound the attention-kernel contribution
    model2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    with torch.no_grad():
        hf2 = torch.log_softmax(model2(ids.cuda()).logits[0].float(), -1).cpu()
    d2 = [abs(hf2[t - 1, ids[0, t].item()].item() - hf_logp[t - 1, ids[0, t].item()].item()) for t in vl]
    print(f"HF eager vs HF sdpa: mean |Δlogp|={sum(d2)/len(d2):.4f} max={max(d2):.4f}")
if __name__ == "__main__":
    main()
