"""Check that the patched extract_solution scores a boxed last-line response (no 'Assistant:' marker)
through the `sec` reward manager's scorer selection, for each benchmark / split."""
import json
from verl.workers.reward_manager.sec import default_compute_score
from verl.utils.reward_score import extract_solution

cases = [
    ("countdown-easy-train", "Let me think.\nStep 1 ...\nThe answer is \\boxed{(7+3)*2}", {"target": 20, "numbers": "7,3,2"}, 1.0),
    ("countdown-hard-test",  "wrong\n\\boxed{7+3+2}",                                      {"target": 20, "numbers": "7,3,2"}, 0.0),
    ("arc-medium-train",     "reasoning here\n\\boxed{0 0 1 1 0}",                         {"target": "0 0 1 1 0"}, 1.0),
    ("zebra-very_hard-test", "reasoning here\n\\boxed{Alice}",                              {"target": "alice"}, 1.0),
    ("arc-easy-test",        "no answer given at all",                                      {"target": "0 0 1 1 0"}, 0.0),
]
for src, resp, gt, want in cases:
    got = default_compute_score(src, resp, gt, None)
    got = got["score"] if isinstance(got, dict) else got
    print(f"{src:22s} extract={extract_solution(resp)!r:14s} score={got} (want {want})")
    assert float(got) == want, (src, got, want)
print("OK")
