"""Check CUDA RNG primitives used by vLLM's native sampler (exponential_ Gumbel-max) on this GPU."""
import torch, collections
def main():
    dev = "cuda"
    print("device", torch.cuda.get_device_name(0), torch.__version__)
    # 1) exponential_ zeros / min / repeats
    for n in [151936, 151936 * 64]:
        q = torch.empty(n, device=dev).exponential_()
        print(f"exponential_ n={n}: zeros={(q == 0).sum().item()} min={q.min().item():.3e} max={q.max().item():.3e} nan={torch.isnan(q).sum().item()}")
    q1 = torch.empty(151936, device=dev).exponential_(); q2 = torch.empty(151936, device=dev).exponential_()
    print("consecutive exponential_ draws identical?", bool(torch.equal(q1, q2)), " first vals", q1[:3].tolist(), q2[:3].tolist())
    g = torch.Generator(device=dev); g.manual_seed(1)
    a = torch.empty(151936, device=dev).exponential_(generator=g); b = torch.empty(151936, device=dev).exponential_(generator=g)
    print("seeded generator consecutive draws identical?", bool(torch.equal(a, b)))
    r1 = torch.rand(8, device=dev); r2 = torch.rand(8, device=dev); print("rand consecutive identical?", bool(torch.equal(r1, r2)))
    u = torch.rand(151936 * 64, device=dev); print(f"rand: zeros={(u==0).sum().item()} ones={(u==1).sum().item()} min={u.min().item():.3e}")
    # 2) random_sample fidelity (vLLM's native sampler)
    V = 151936
    probs = torch.full((1, V), 1e-6, device=dev); probs[0, 271] = 1.0; probs /= probs.sum()
    print("dominant prob =", probs[0, 271].item())
    wrong = collections.Counter(); N = 4000
    for _ in range(N):
        q = torch.empty_like(probs).exponential_()
        tok = (probs / q).argmax(dim=-1).item()
        if tok != 271: wrong[tok] += 1
    print(f"vLLM random_sample: {sum(wrong.values())}/{N} wrong picks (expected ~{N*(1-probs[0,271].item()):.1f}); top wrong ids: {wrong.most_common(5)}")
    # same with multinomial
    picks = torch.multinomial(probs.expand(N, V), 1).squeeze(-1)
    print(f"torch.multinomial: {(picks != 271).sum().item()}/{N} wrong picks")
    # 3) masked (top-p style) distribution: zeros everywhere except 3 tokens
    probs2 = torch.zeros((1, V), device=dev); probs2[0, 271] = 0.8; probs2[0, 198] = 0.15; probs2[0, 13] = 0.05
    wrong2 = collections.Counter()
    for _ in range(N):
        q = torch.empty_like(probs2).exponential_()
        tok = (probs2 / q).argmax(dim=-1).item()
        if tok not in (271, 198, 13): wrong2[tok] += 1
    print(f"masked dist random_sample: {sum(wrong2.values())}/{N} picks outside support; top ids: {wrong2.most_common(5)}")
    # 4) CPU reference
    qc = torch.empty(151936 * 64).exponential_(); print(f"CPU exponential_: zeros={(qc==0).sum().item()} min={qc.min().item():.3e}")
    # 5) bf16/fp32 softmax + argmax sanity on GPU vs CPU for a random logits row
    torch.manual_seed(0); lg = torch.randn(4, V) * 3
    print("argmax gpu==cpu:", bool(torch.equal(lg.cuda().argmax(-1).cpu(), lg.argmax(-1))), " softmax max abs diff:", (lg.cuda().softmax(-1).cpu() - lg.softmax(-1)).abs().max().item())
if __name__ == "__main__":
    main()
