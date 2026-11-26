"""Is the 'garbage token' effect a vLLM/Blackwell bug or normal T=1.0 sampling of a base model?
(1) compare next-token distributions HF(sdpa, bf16) vs vLLM on prompt positions (prompt_logprobs);
(2) sample with HF at T=1 p=1 / p=0.7 and count weird chars like the vLLM run;
(3) tail mass of the model's distribution along a greedy path."""
import os, re, math, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
def weird(s): return len(re.findall(r"[^\x00-\x7F]", s))
def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    q = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
         "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
         "User: Let $p(x)$ be a quadratic polynomial with integer coefficients which has $4-\\sqrt{11}$ as a root. "
         "Compute $\\frac{p(3)}{p(4)}.$ Put your final answer within \\boxed{}. \nAssistant: Let me solve this step by step.")
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    with torch.no_grad():
        logits = model(ids).logits[0].float()           # [T, V]
    hf_logp = torch.log_softmax(logits, -1)
    # greedy continuation with HF + tail mass stats
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, do_sample=False, output_scores=True, return_dict_in_generate=True)
    tails10, tails100 = [], []
    for sc in out.scores:
        p = torch.softmax(sc[0].float(), -1); top = p.topk(100).values
        tails10.append(1 - top[:10].sum().item()); tails100.append(1 - top.sum().item())
    print(f"HF greedy: weird={weird(tok.decode(out.sequences[0, ids.shape[1]:]))}  mean P(outside top10)={sum(tails10)/len(tails10):.4f}  mean P(outside top100)={sum(tails100)/len(tails100):.4f}  max P(outside top100)={max(tails100):.4f}")
    # HF sampling
    for name, kw in {"T1_p1.0": dict(temperature=1.0, top_p=1.0), "T1_p0.7": dict(temperature=1.0, top_p=0.7)}.items():
        torch.manual_seed(0)
        with torch.no_grad():
            o = model.generate(ids.expand(4, -1), max_new_tokens=300, do_sample=True, top_k=0, **kw)
        for i in range(4):
            t = tok.decode(o[i, ids.shape[1]:], skip_special_tokens=True)
            print(f"HF {name} #{i}: weird={weird(t)} :: {t[:300]!r}")
    del model; torch.cuda.empty_cache()
    # vLLM prompt logprobs vs HF
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-3B", gpu_memory_utilization=float(os.environ.get("GMU", "0.3")), max_model_len=2048, dtype="bfloat16", seed=1, enable_prefix_caching=False)
    o = llm.generate([prompt], SamplingParams(max_tokens=1, prompt_logprobs=0, logprobs=20, temperature=0))[0]
    # prompt_logprobs: list (len T) of dict(token_id -> Logprob) for the *actual* next token; compare logprob of actual tokens
    diffs = []
    for t in range(1, ids.shape[1]):
        d = o.prompt_logprobs[t]
        if not d: continue
        tid = ids[0, t].item()
        if tid in d:
            diffs.append(abs(d[tid].logprob - hf_logp[t - 1, tid].item()))
    print(f"vLLM vs HF logprob of actual prompt tokens: n={len(diffs)} mean abs diff={sum(diffs)/len(diffs):.4f} max={max(diffs):.4f}")
    # next-token top-20 at the end of the prompt
    last = o.outputs[0].logprobs[0]
    hf_top = torch.topk(hf_logp[-1], 20)
    print("HF top5 next tokens:", [(tok.decode([i]), round(v.item(), 3)) for v, i in zip(hf_top.values[:5], hf_top.indices[:5])])
    print("vLLM top5 next tokens:", sorted([(tok.decode([k]), round(v.logprob, 3)) for k, v in last.items()], key=lambda x: -x[1])[:5])
    kl = 0.0
    for k, v in last.items():
        kl += math.exp(v.logprob) * (v.logprob - hf_logp[-1, k].item())
    print(f"KL(vLLM||HF) over vLLM top-20 next tokens: {kl:.5f}")
    # vLLM sampling rates with more samples
    for name, sp in {"T1_p1.0": SamplingParams(temperature=1.0, top_p=1.0, max_tokens=300, n=8, seed=3), "T1_p0.7": SamplingParams(temperature=1.0, top_p=0.7, max_tokens=300, n=8, seed=3)}.items():
        outs = llm.generate([prompt], sp)[0].outputs
        ws = [weird(x.text) for x in outs]; toks = sum(len(x.token_ids) for x in outs)
        print(f"vLLM {name}: weird chars per sample={ws} total tokens={toks} rate={sum(ws)/toks:.4f}")
if __name__ == "__main__":
    main()
