"""Diagnose corrupted validation generations (garbage tokens where '\\n\\n' / EOS should be) on Blackwell.
Runs vLLM on Qwen2.5-3B with greedy / top_p=1.0 / top_p=0.7 and tests vLLM's native top-p op directly."""
import os, sys, re, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")

def weird(s):  # count non-ASCII, non-LaTeX-ish chars in an English math answer
    return len(re.findall(r"[^\x00-\x7F]", s))

def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    q = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
         "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
         "User: Let $p(x)$ be a quadratic polynomial with integer coefficients which has $4-\\sqrt{11}$ as a root. "
         "Compute $\\frac{p(3)}{p(4)}.$ Put your final answer within \\boxed{}. \nAssistant: Let me solve this step by step.")
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    llm = LLM(model="Qwen/Qwen2.5-3B", gpu_memory_utilization=float(os.environ.get("GMU", "0.25")), max_model_len=3072, dtype="bfloat16", seed=1)
    cases = {
        "greedy": SamplingParams(temperature=0, max_tokens=400),
        "T1_p1.0": SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400, seed=1),
        "T1_p0.7": SamplingParams(temperature=1.0, top_p=0.7, max_tokens=400, seed=1),
        "T1_p0.7_k50": SamplingParams(temperature=1.0, top_p=0.7, top_k=50, max_tokens=400, seed=1),
        "T0.6_p0.95": SamplingParams(temperature=0.6, top_p=0.95, max_tokens=400, seed=1),
    }
    for name, sp in cases.items():
        outs = llm.generate([prompt] * 3, sp)
        for i, o in enumerate(outs):
            t = o.outputs[0].text
            print(f"=== {name} #{i}: len={len(o.outputs[0].token_ids)} weird_chars={weird(t)} finish={o.outputs[0].finish_reason}")
            print(t[:500].replace("\n", "⏎"))
        sys.stdout.flush()

    # direct test of vLLM's native top-p op on a peaked distribution
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    V = len(tok)  # 151665 real vocab; also test padded 151936
    for vocab in [V, 151936]:
        torch.manual_seed(0)
        logits = torch.randn(4, vocab, device="cuda", dtype=torch.float32)
        logits[:, 271] = 25.0  # '\n\n' extremely likely
        logits[0, 151645] = 30.0  # EOS even more likely in row 0
        out = apply_top_k_top_p(logits.clone(), None, torch.full((4,), 0.7, device="cuda"))
        kept = (out > -float("inf")).sum(dim=-1).tolist()
        am = out.argmax(dim=-1).tolist()
        print(f"apply_top_k_top_p vocab={vocab}: kept per row={kept} argmax={am} (expect [151645,271,271,271])")
        probs = torch.softmax(out, dim=-1)
        samp = torch.multinomial(probs, 1).squeeze(-1).tolist()
        print(f"  sampled ids={samp} -> {[tok.decode([s]) for s in samp]!r}")
    # torch.sort / scatter sanity on sm_120
    x = torch.randn(2, 151936, device="cuda")
    s, idx = x.sort(dim=-1, descending=False)
    back = torch.empty_like(s).scatter_(dim=-1, index=idx, src=s)
    print("sort/scatter roundtrip max abs err:", (back - x).abs().max().item(), " sorted ok:", bool((s[:, 1:] >= s[:, :-1]).all()))
    cs = torch.cumsum(s.softmax(-1), dim=-1); cs_cpu = torch.cumsum(s.softmax(-1).cpu(), dim=-1)
    print("cumsum gpu vs cpu max abs err:", (cs.cpu() - cs_cpu).abs().max().item())
    p = s.softmax(-1); torch.cumsum(p, dim=-1, out=p)  # in-place as vLLM does
    print("in-place cumsum vs cpu max abs err:", (p.cpu() - cs_cpu).abs().max().item())

if __name__ == "__main__":
    main()
