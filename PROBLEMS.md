# PROBLEMS.md — questions, decisions and problems from the end-to-end reproduction

Reproduction of arXiv:2602.20532 (Actor-Curator) Table 1 following `README.md`.
Written by Claude while running the reproduction on the titan cluster (2026-11-04).
Each item is either a **DECISION** I made (with the assumption I proceeded under) or a **QUESTION / PROBLEM**
that needs your input. Nothing in `verl/` or `recipe/` was modified; all deviations are passed as Hydra
overrides from wrapper scripts in `repro/` (new, untracked directory).

---

## A. Environment / cluster (problems I hit and how I worked around them)

### A1. No Docker on this cluster; GPUs are RTX PRO 6000 Blackwell (sm_120), not A100/H200
- README recommends the verl 0.5 Docker image (cu126). Docker/Apptainer are not available on titan, and the
  cu126 stack has no sm_120 kernels anyway.
- **DECISION:** built a conda env `verl` (Python 3.12) that mirrors the repo's own cu128 recipe
  (`docker/verl0.5-preview-cu128-torch2.7.1-fa2.8.0/Dockerfile.base`): torch 2.7.1+cu128, vLLM 0.10.0,
  flash-attn 2.8.0.post2 (cu12/torch2.7 wheel), transformers pinned by this fork's `setup.py` (<4.54), `pip install -e .[test,math]`.
  Install script: `repro/install_env.sh`. Any numerical difference vs. the paper's A100/H200 runs is possible
  (bf16 kernels differ) but should be within run-to-run noise.
- Dependency fixes needed on top of the repo's pins (`numpy<2`): `pyarrow==20.0.0` (newer needs numpy 2),
  `scipy<1.16` (same), `matplotlib<3.11` (imported unconditionally by `ray_trainer.py`, not in any requirements file).
- Verified on a titan GPU: torch 2.7.1+cu128 sees sm_120, flash-attn 2.8.0.post2 forward+backward works, vLLM 0.10.0 imports.

### A2. GPU budget
- My SLURM QoS (`normal`, account `raul`) caps me at **4 GPU-equivalents** concurrently (preemptable) +
  1 guaranteed GPU (`guaranteed-raul`, not usable for 2-GPU jobs). README says the paper used 2×A100-80GB per 3B run.
- First plan was `--gres=gpu:2` per run (README setup). **Problem:** on titan a `gpu:2` allocation does not give
  exclusive cards — my first smoke job was placed on two cards that other users' shard jobs were already using
  (43 GB and 14 GB occupied), which breaks vLLM's `gpu_memory_utilization=0.85` reservation. The cluster guide says
  only `--gres=shard:10` (one whole card) is "the safer spelling" for exclusive use.
- **DECISION:** run every cell on **one whole card** (`shard:10`, `trainer.n_gpus_per_node=1`, `gen_tp=1`; batch
  sizes, mini-batches and all algorithmic knobs unchanged — DP=1 instead of DP=2 is mathematically the same update,
  only slower per step). This allows up to 4 concurrent runs under `normal` plus 1 under `guaranteed-raul`.
  The cluster is saturated (all 80 shards allocated on titan-3, RAM fully allocated on titan-1, 144 pending jobs);
  I use the group's guaranteed 1-GPU quota (`--qos=guaranteed-raul`) for one run at a time, which can pre-empt
  (requeue) other users' `normal` jobs as per cluster policy. → QUESTION: OK to keep using `guaranteed-raul`?
  If you can get `normal-plus` (6 GPUs) tell me.

### A4. HF `datasets` cache in the shared model cache (fixed)
- With `HF_HOME=/shared/models/huggingface` (cluster convention) the `datasets` library writes its parquet cache
  there; that tree chowns new files to `shared-data`, so `datasets`' `os.chmod` fails with `PermissionError`.
  **Fix:** `HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets` in `repro/run.sh` (no code change).

### A5. SLURM device variables (fixed)
- SLURM exports `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`/`GPU_DEVICE_ORDINAL` next to `CUDA_VISIBLE_DEVICES`;
  verl's worker raises `ValueError: Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set`.
  **Fix:** the wrapper unsets the three ROCm variables before launching (no code change).

### A6. Node mbz-titan-3: NFS client hangs on files of my conda env (open)
- The first real run (job 72504, countdown AC) sat 40 min on titan-3 with no output: the driver was in kernel
  state `nfs_wait_bit_killable` (D). Outside SLURM, `python -B -c "import numpy"` from the `verl` env also hangs on
  titan-3 (right after `numpy.core.defchararray`), while plain python, `touch`, and 1 GB/s `dd` reads of other env
  files work there, and the same imports work on titan-1/titan-2. Looks like stale NFS client state on titan-3
  for a few files (env was created from titan-1 at 08:2x). → **Admin may need to drop caches / remount `/shared`
  on titan-3.** Workaround: all my pending jobs now carry `--exclude=mbz-titan-3`; countdown AC resubmitted.

### A7. "Corrupted" validation generations — investigated, NOT a bug of this GPU stack (resolved)
- Logged validation samples (`val/generations` in TensorBoard) of the first MATH AC run contained odd tokens where
  structural tokens belong (`' gag'`, `'Ërgo'` instead of `'\n\n'`, `'虼'` instead of `<|im_end|>`, after which the
  base model keeps copying the glitch and never stops). Step-0 MATH500 accuracy was 16.8 % (paper π_ref 61.8 %).
- Diagnostics (jobs 72570, 72578, 72580; scripts `repro/*_check.py`):
  - vLLM's native top-p op, sort/scatter/cumsum, `exponential_` Gumbel-max sampler: all numerically correct on sm_120.
  - vLLM greedy output is clean; the glitches appear only with sampling at temperature 1.0.
  - **HF transformers (sdpa, bf16) sampling at T=1.0 shows the same glitches at the same rate** (e.g. `𫓯`, `榅`,
    `ogłoszeniAssistant`); the base model's distribution has ~0.3 % mass outside the top-100 tokens per step (up to 23 %
    at some positions), so at T=1.0 a glitch token is drawn every few hundred tokens and the base model then copies it.
  - Direct vLLM-vs-HF logprob comparison of the forward pass (job 72584): mean |Δlogp| 0.030, KL 0.006 on the next-token
    distribution — same size as HF eager-vs-sdpa differences (0.037), i.e. bf16 noise. The forward pass is correct on sm_120.
- Conclusion: this is the normal behaviour of the Qwen2.5-3B *base* model under the paper's sampling settings
  (train T=1.0/top-p 1.0, validation T=1.0/top-p 0.7); RL training is expected to suppress it. Runs released again.
- → QUESTION: the paper's π_ref for MATH500 (61.80 %) is far above what the base model scores here at step 0 (16.8 %)
  with the repo's prompt/verifier/sampling. Do you know how π_ref was evaluated (greedy? instruct prompt?)? Countdown
  π_ref = 0.00 matches.

- 2026-11-26 UPDATE: greedy decoding does NOT close the gap. Job 96517 evaluated the *untrained* Qwen2.5-3B with `do_sample=false` on the recipe's prompt (R1-style "A conversation between User and Assistant …", `\boxed{}` instruction): MATH500 **26.8 %**, AMC 15.7 %, AIME24 0.0 % (sampled T=1.0/top-p 0.7 at step 0: 16.8 / 9.6 / 0.0). So decoding temperature explains ≤10 of the 45 points to the paper's π_ref 61.8. What is left: a different prompt/few-shot or chat-template evaluation, a different grader, or a different checkpoint (as I recall, the Qwen2.5 report lists the 3B *base* at ≈42.6 % MATH 4-shot and the 3B *Instruct* at ≈65.9 %). Our trained MATH runs plateau at 64–65 %, i.e. about the paper's π_ref. → still a QUESTION for you; nothing to change on our side without knowing the paper's protocol.
### A8. Countdown / Zebra / ARC reward is identically zero with the shipped code  **(RESOLVED 2026-11-22 — patch applied on your instruction)**
- `verl/utils/reward_score/__init__.py::extract_solution` (used by the countdown/zebra/arc scorers) first cuts the text at
  an `"Assistant:"` / `"<|im_start|>assistant"` marker and **returns `None` if the marker is absent**; the `sec` reward
  manager (`verl/workers/reward_manager/sec.py`, used by every script) passes the decoded **response only**, which never
  contains the marker (it is in the prompt). Result: every well-formed response scores 0, in training and validation.
  Verified directly: `arc_score("...\n\\boxed{6 0 6 ...}", gt)` → 0, the same string prefixed with `"Assistant: "` → 1.0
  (same for zebra). In the ARC AC run the scorer's own debug prints showed 341× "No equation found", 1× "Wrong result"
  (a degenerate response that echoed "Assistant:"), 0× correct; mean training reward ≈ 0.001 after 10 steps.
  The MATH scorer (`deepscaler`, used for `math_train` / `test_*`) does not use this extractor → MATH runs are fine.
- The marker logic is TinyZero's, written for a reward manager that decodes prompt+response together; this fork's
  managers decode the response alone (since the first commit of these files, 2025-09-29), so the paper's Countdown /
  Zebra / ARC numbers cannot have been produced by the code as it is in this repo.
- **Proposed 1-line fix** (not applied; patch in `repro/fix_extract_solution.patch`): in `extract_solution`, replace
  `else: return None` by `else: pass`, i.e. score the response as-is (the rest — last line, last `\boxed{}` — unchanged).
  Alternative with identical effect: pass `prompt_str + response_str` from `sec.py`.
- Actions taken: ARC AC run cancelled, all Countdown/Zebra/ARC jobs (incl. their smoke tests) put on hold;
  MATH AC resubmitted on the guaranteed slot, MATH baselines left in the `normal` queue.
- → QUESTION: may I apply the patch (`git apply repro/fix_extract_solution.patch`) and release the held runs?
- **DECISION (you, 2026-11-22): "apply the patch and relaunch the jobs".** Applied with `git apply repro/fix_extract_solution.patch`
  (the only change under `verl/`: `git diff verl/utils/reward_score/__init__.py`). Verified with `repro/scorer_check.py`
  (patched extractor through the `sec` reward manager's routing): countdown/arc/zebra boxed last-line answers → 1.0, wrong
  equation → 0.0, no box → 0. Note the shipped behaviour kept as-is: only the **last line** of the response is scanned for
  the last `\boxed{}` (TinyZero convention); a response that continues after the box scores 0.

### A9. SEC script crashes at step 1: `Key 'temperature' is not in struct` (fixed by an override)
- `countdown_topk.sh` (the only clean 3B SEC script) does not set `+adv_predictor.temperature`, but `ray_trainer.py`
  reads `config.adv_predictor.temperature` unconditionally every step, so the first MATH SEC attempt (73339) crashed
  right after the step-0 validation. The `stochastic_topk` sampler does not use the value.
  **Fix:** the wrapper now passes `++adv_predictor.temperature=1.0` for SEC runs (no code change); MATH SEC resubmitted.
- Same class of problem for PCL: `countdown_ps.sh` / `zebra_ps_value.sh` / `arc_ps_value.sh` never set
  `+adv_predictor.target`, but `ray_trainer.py` evaluates `config.adv_predictor.target == "perf_diff"` unconditionally
  → MATH PCL (72513) crashed at step 1 with `Key 'target' is not in struct`. **Fix:** wrapper passes
  `++adv_predictor.target=abs_adv` for PCL (the README's PCL definition); MATH PCL resubmitted (new job, queue age lost).

### A3. Disk quota (500 GB home, 336 GB already used) vs. checkpoints
- A 3B-actor FSDP checkpoint (+0.6B curator, fp32 optimizer state) is ~40 GB. The scripts save every 50 steps and keep all.
- **DECISION:** checkpoints go to node-local `/scratch/$USER/actor-curator-ckpt/...` (not in quota) with
  `trainer.save_freq=20`, `max_actor_ckpt_to_keep=1`, `max_critic_ckpt_to_keep=1`, `trainer.resume_mode=auto`,
  so a pre-empted/requeued job can resume if it lands on the same node (otherwise it restarts from step 0).
  Note: the curator replay buffer / per-problem history are not part of verl checkpoints, so a resumed AC run is
  not bit-identical to an uninterrupted one. Final metrics come from TensorBoard logs in `tensorboard_log/`, which are in the repo dir.

---

## B. Protocol decisions (paper vs. shipped scripts)

The README says the method is defined by config knobs, but the shipped `recipe/deepscaler/*.sh` values differ
from the paper's Table 2 (App. G.1) in several places, and differ from each other. The `exp_name`s inside the
scripts (`..._temp_1_topp_0.9_...`, `..._cliprange_0.2_...`, `..._amplifier_1000`) suggest the paper runs used
the Table-2 values and the scripts were edited afterwards for exploratory runs.

### B1. Curator dormant / warm-up steps  **(important)**
- Paper Table 2: dormant = 20, warm-up = 5. Scripts: countdown_perf_diff 100/20, arc_perf_diff 30/5,
  zebra/math_perf_diff 20/5, countdown_topk (SEC) 70/5, countdown_ps / zebra_ps_value / arc_ps_value (PCL) 70/5 / 50/5 / 70/5.
- In the code the curator only influences selection when `global_step > dormant + warmup`, so with the shipped
  Countdown AC script (120) the curator would **never act within the 100-step evaluation window** → it would be
  identical to Uniform. Same problem (steps 76–100 only) for the shipped SEC/PCL scripts.
- **DECISION:** use the paper's 20/5 for **all** learned methods (AC, SEC, PCL) on all benchmarks.
  → QUESTION: do you agree, or should baselines keep their script values (70/5)?

### B2. PCO clip range
- Paper Table 2: 0.2. Scripts: countdown/math 0.5, zebra/arc 0.2. **DECISION:** 0.2 everywhere.

### B3. Curator temperature / top-p
- Paper Table 2: temperature 1.0, top-p 0.9, no annealing. Scripts: zebra_perf_diff anneals temperature 5→2,
  arc_perf_diff uses 2.0; PCL scripts anneal top-p 0.9→0.7 over `total_training_steps`.
- **DECISION:** temperature 1.0, top-p 0.9, annealing off for all runs (for PCL this also avoids the annealing
  schedule collapsing when `total_training_steps` is reduced to 100, see B5).

### B4. Knobs the paper does not specify (kept from scripts)
- `perf_diff_amplifier`: countdown/math scripts 1000, zebra/arc 100 (arc's exp_name says 1000). **DECISION:** 1000 everywhere.
- `use_window_avg_target/history_length`: zebra & math scripts true/10, countdown & arc false. **DECISION:** kept per script.
- `replay_buffer_size=2`, `adv_predictor.train_batch_size=256`, `ema_coeff` (only used by an unrelated
  difficulty-curriculum path), `use_sampling_prior=false`: kept.
- PCL (`value_head`): `num_heads=2, loss_fn2_weight=1.0, history_length=3` kept from `countdown_ps.sh`.

### B5. Training length and LR warm-up
- Paper reports the peak validation accuracy within the first 100 steps (eval every 10 steps) of longer runs
  (figures go to 500 steps). Scripts set `total_training_steps` to 500/600/1000/2000/10,000,000 (!) with
  `lr_warmup_steps_ratio=0.05`, so the actor's warm-up length ranges from 25 to 500k steps depending on the script
  (`countdown_baseline.sh` as shipped would effectively train at lr≈0 for its first 100 steps).
- **DECISION:** every run trains for exactly **100 steps** (`trainer.total_training_steps=100`, `test_freq=10`,
  `val_before_train=true`) with an explicit **`lr_warmup_steps=25`** (= 0.05 × 500, i.e. the schedule of a
  500-step run as in the paper's figures). → QUESTION: OK? Alternative is ratio 0.05 of 100 = 5 steps, or running the
  full 500 steps (5× the GPU cost) to also reproduce the training curves.

### B6. Which script backs each (benchmark, method) cell
| Benchmark | AC | Uniform | SEC | PCL |
|---|---|---|---|---|
| Countdown | `countdown_perf_diff.sh` | `countdown_baseline.sh` | `countdown_topk.sh` | `countdown_ps.sh` |
| Zebra | `zebra_perf_diff.sh` (MODEL→3B) | `zebra_baseline.sh` | **`countdown_topk.sh` + zebra data** | `zebra_ps_value.sh` |
| ARC-1D | `arc_perf_diff.sh` (MODEL→3B) | `arc_baseline.sh` (MODEL→3B) | **`countdown_topk.sh` + arc data** | `arc_ps_value.sh` |
| MATH | `math_perf_diff.sh` (MODEL→3B) | `math_baseline.sh` (MODEL→3B) | **`countdown_topk.sh` + math data** | **`countdown_ps.sh` + math data** |
- Bold cells have no clean 3B script in the repo (README "Known gaps"); **DECISION:** reuse the Countdown
  script of that method with `data.train_files/val_files` pointed at the other benchmark (the method is defined
  by knobs, not by the filename). `zebra_topk.sh`/`math_topk.sh` ship with Llama-3.2-3B-Instruct and
  `arc_topk.sh` has an empty `MODEL_PATH=Qwen/`, so they were not used.
- `MODEL_PATH` is changed to `Qwen/Qwen2.5-3B` by `sed` into a temporary copy of the script (the repo file is untouched).

### B7. Hardware-related knobs
- `GPUS_PER_NODE` is set to 2 (several scripts use `nvidia-smi --list-gpus | wc -l`, which would be 8 on titan).
  `gen_tp` is left as in each script (countdown AC uses TP=2, others TP=1).
- `WORKING_DIR` is overridden to this checkout.
- Single-card runs set `actor_rollout_ref.rollout.gpu_memory_utilization=0.75` (scripts: 0.85). With actor, curator
  and vLLM on one 96 GB card ~15 GB are resident when vLLM starts, so 0.85 (80.7 GiB) does not fit. This only
  changes vLLM's KV-cache size (rollout throughput), not the algorithm.

---

## C. Observations about the repo/paper that you may want to know
- README's "App. K" (7B / Qwen3-1.7B) does not exist in the arXiv paper (appendices go A–J; the only alternative
  actor is Llama-3.2-3B-Instruct, Table 4).
- The repo's "SEC" (`stochastic_topk`) selects the 256 problems whose predicted value is closest to 0.5 (a
  PCL-like rule), and the repo's "PCL" (`value_head` + nucleus `uniform` sampler) samples by softplus score.
  Neither is the bucket-bandit SEC of Chen et al. described in the paper's App. A.3. I ran what the README maps.
- Evaluation: Table 1 numbers are consistent with Countdown/Zebra/ARC test banks of 800 problems plus 200-problem
  "hard" subsets (0.125 % granularity), AIME24 = 30 problems.

### B8. What the shipped test banks contain, and which subset is the paper's "-hard"  (QUESTION)
- `test_{countdown,zebra,arc}.parquet`: 800 problems = 4 `data_source`s × 200: `<bench>-easy-test`, `-medium-test`,
  `-hard-test`, `-very_hard-test` (train banks only have easy/medium/hard, 10k each → `very_hard` is an unseen level).
- `math_test.parquet`: `test_math` (500 = MATH500), `test_amc` (83), `test_aime` (30 = AIME24). The paper's MATH500 /
  AIME24 columns map to `test_math` / `test_aime`; AMC is evaluated too but not in the paper's tables.
- verl logs one accuracy per `data_source` (`val-core/<data_source>/reward/mean@1`); the "Countdown" column of
  Table 1 (granularity 1/800) must be the mean over the 4 levels, and "Countdown-hard" (granularity 1/200) is one
  level. **DECISION:** report all four levels + their mean, and label `-very_hard-test` as the paper's "-hard" row
  (the held-out hardest level) while also showing `-hard-test`. → QUESTION: which of the two is the paper's "-hard"?
- Validation sampling (from the scripts): 1 sample per problem, temperature 1.0, top-p 0.7 → matches Table 2.

---

- 2026-11-22, re-read of arXiv:2602.20532 (for "why is MATH 17 points below the paper"):
  the paper names the actor **Qwen2.5-3B-Base** (App. J) = what we run; π_ref is "the pretrained model without
  post-training" and its decoding (greedy vs. sampling, temperature, prompt template) is **not stated**; validation
  temperature, samples per problem, test-set sizes and the number of GPUs are not stated either; total training steps
  are not stated (figures/Table 3 mention 200 and 500 steps); single values, no seeds/error bars. Table 1 MATH500:
  π_ref 61.80, Uniform 83.00, SEC 81.00, PCL 79.79, AC 81.00. Our step-0 (= π_ref) MATH500 is 16.8 % with the repo's
  prompt (TinyZero "A conversation between User and Assistant…"), verifier and validation sampling (T=1.0, top-p 0.7,
  1 sample, 4096 tokens). A 45-point gap *before any training* means the measurement differs (decoding/prompt/verifier),
  not the algorithm; the post-training gap (~17 points) is of the same kind. Training set matches (11,499 vs "12K").

## D. Status log
(updated as the reproduction progresses)

- 2026-11-04 08:40 — conda env built and verified on GPU (torch/flash-attn); vLLM check + 3-step smoke tests in progress.
- 2026-11-04 09:16 — single-card smoke test of the full AC pipeline passed (3 steps + validation, job 72488).
- 2026-11-04 09:18 — submitted the 16 real runs: AC runs 72504-72507 on `guaranteed-raul` (sequential), baselines
  72508-72519 on `normal`. Smoke tests for Uniform/SEC/PCL (72489-72491) still queued ahead of them.
  Progress is appended to `repro/logs/watch.log`; results via `python repro/collect_results.py`.
- 2026-11-04 10:30 — garbage-token scare resolved (A7); all runs released again (AC chain on guaranteed-raul, baselines on normal, titan-3 excluded).
- 2026-11-04 10:45 — ARC AC (72506) step time on one card: 685 s/step (rollout 220 s, actor update 317 s with FSDP CPU
  offload, curator 83 s) → ~20 h per 100-step run incl. validations. AC column via the guaranteed chain ≈ 3.5 days;
  baselines depend on the `normal` queue (age-ordered, 140+ jobs ahead). Option to speed up ~20 %: disable FSDP
  param/optimizer offload (hardware knob) at some OOM risk on 96 GB — not done.
- 2026-11-04 12:35 — A8 found: reward extractor returns None without an 'Assistant:' marker → Countdown/Zebra/ARC rewards are 0. Those 14 jobs held; MATH AC (72665) queued on guaranteed; waiting for your go-ahead on the 1-line fix.
- 2026-11-05 00:30 — MATH AC (72665) validation so far (MATH500 / AMC / AIME24, %): step0 16.8/9.6/0 · 10 22.0/7.2/0 ·
  20 43.2/21.7/3.3 · 30 57.2/26.5/3.3 · 40 61.4/32.5/0 · 50 62.6/26.5/6.7 · 60 63.4/28.9/3.3 · 70 63.2/30.1/6.7 ·
  80 63.6/38.6/10.0 · 90 63.0/28.9/16.7 (paper AC peak: 81.0 / — / 30.0). MATH Uniform (72511) started 23:57 on titan-2.
  MATH SEC moved from the `normal` queue to the guaranteed chain (73339) so the slot is not idle after MATH AC ends.
- 2026-11-05 01:20 — **MATH AC (72665) finished** (100 steps, 12.3 h on one card). Peak within 100 steps:
  MATH500 **64.2 %** (step 100), AMC 38.6 % (step 80), AIME24 **16.7 %** (step 90). Paper (AC): MATH500 81.0, AIME24 30.0.
  Step-100 values: 64.2 / 33.7 / 10.0. Final checkpoint kept at /scratch/kseniia.kuvshinova/actor-curator-ckpt/math_ac_qwen2.5-3b_s1/global_step_100 (titan-1).
  MATH SEC (73339) started on the guaranteed slot right after; MATH Uniform (72511) running on titan-2.
- 2026-11-05 02:50 — MATH Uniform (72511, `normal` QoS) was pre-empted at step 17 (another group's guaranteed job took
  the card); no checkpoint yet (save every 20 steps) → it will restart from step 0 when it gets a card again. MATH PCL was
  pre-empted at step ~1 by my own guaranteed MATH SEC job. Pre-emption losses are inherent to the `normal` queue; checkpoints
  are node-local, so even with them a restart on another node begins at step 0.
- 2026-11-05 08:00 — MATH SEC (73446) at step 79: val MATH500/AMC/AIME (%) step10 26.0/4.8/0 · 20 43.6/18.1/3.3 · 30 60.6/21.7/3.3 ·
  40 63.6/31.3/10.0 · 50 59.2/30.1/10.0 · 60 64.4/37.3/3.3 · 70 62.8/30.1/3.3. MATH Uniform restarted 06:22 (step 10: 26.8/13.3/3.3).
  MATH PCL resubmitted as 73730 (queued). Watcher fixed to rescan logs truncated by a requeue.
- 2026-11-05 08:15 — MATH Uniform (72511) pre-empted a second time (step ~13, again before its first checkpoint).
  DECISION: moved it to the guaranteed chain (new job queued behind MATH SEC); MATH PCL (73730) stays in `normal` for now.
  Net effect: the `normal` queue has not completed a single run in 23 h; the guaranteed slot is the only reliable capacity.
- 2026-11-05 09:00 — **MATH SEC (73446) finished** (7.5 h). Peak ≤100 steps: MATH500 **64.4 %** (step 60), AMC 37.3 % (60),
  AIME24 **10.0 %** (steps 40/50/90). Step 100: 63.6 / 30.1 / 6.7. Paper (SEC): MATH500 81.0, AIME24 20.0.
  MATH Uniform (73859) now starts on the guaranteed slot; MATH PCL (73730) running under `normal` since 08:24.
- 2026-11-05 19:50 — **MATH row complete** (Qwen2.5-3B, 100 steps, eval every 10, peak within 100 steps, %):

  | | Uniform | SEC | PCL | AC |
  |---|---|---|---|---|
  | MATH500 ours | 64.2 (step 100) | 64.4 (60) | 65.4 (100) | 64.2 (100) |
  | MATH500 paper | 83.0 | 81.0 | 79.8 | 81.0 |
  | AIME24 ours | 10.0 (70/80) | 10.0 (40/50/90) | 6.7 (30+) | 16.7 (90) |
  | AIME24 paper | 23.3 | 20.0 | 23.3 | 30.0 |
  | AMC ours | 33.7 | 37.3 | 37.3 | 38.6 |

  All four methods end within 1.2 points of each other on MATH500 (single-sample eval, 500 problems: ±2 points is noise)
  and ~17 points below the paper; AC has the best AIME24 (5/30 vs 2–3/30; 30 problems → one problem = 3.3 points).
  MATH Uniform/PCL each lost one or two restarts to pre-emption; final runs were uninterrupted. Idle now: everything
  else is held pending A8 (reward-extractor patch).
- 2026-11-22 17:00 — Status check: no Actor-Curator jobs running or queued. The 14 held Countdown/Zebra/ARC jobs were
  cancelled by root on 2026-11-07 (SLURM reason JobHeldUser / QOSGrpBilling — admin cleanup of long-held jobs), so they
  must be resubmitted via `repro/submit.sh` once A8 is decided. Patch `repro/fix_extract_solution.patch` still NOT applied
  (`git apply --check` passes). The log watcher (repro/watch.sh) is no longer running; it will be restarted with the next submission.
  Still waiting on: A8 (apply the 1-line reward-extractor fix?), B1, B5, B8, A2, A7.
- 2026-11-22 17:05 — **A8 approved and applied; Countdown/Zebra/ARC relaunched.** The 14 held jobs had been cancelled by
  root on 2026-11-07, so everything was resubmitted with the current wrapper (SEC/PCL overrides from A9 included):
  smoke test 93835 (countdown AC, guaranteed slot, 3 steps — confirms non-zero reward end to end);
  AC runs on `guaranteed-raul` (one after another): 93836 countdown, 93837 zebra, 93838 arc;
  baselines on `normal`: 93839/93840/93841 countdown uniform/sec/pcl, 93842–93844 zebra, 93845–93847 arc.
  titan-3 is back in the node pool (your other jobs ran there fine since 2026-11-20 and `import torch` from the `verl` env
  works there again). `normal` QoS per-user cap is now gres/gpu=3 (was 4 GPU-equivalents), so at most 3 baselines run at once.
  Stale state from the zero-reward runs: TensorBoard dir of ARC AC 72506 moved to repro/logs/stale_tb_arc_ac_zero_reward_job72506
  (no checkpoint existed: it stopped at step 10, save every 20). Expected wall time: ~12 h (countdown/zebra) to ~20 h (arc)
  per run on one card → the AC chain alone needs ~2 days; baselines depend on the `normal` queue.
- 2026-11-22 17:15 — **Smoke test 93835 passed with the A8 patch**: Countdown reward is no longer identically zero
  (val step 0: medium 0.125 (1/8); step 3: easy 0.125; training reward mean step 1: 0.0156 = 1/64 responses correct;
  scorer debug prints now show "Wrong result" (= boxed but wrong, format reward) instead of only "No equation found").
  Real runs started on titan-3: Countdown AC 93836 (guaranteed) 17:15, Countdown Uniform 93839 + SEC 93840 (normal) 17:11.
  Zebra/ARC AC wait on the guaranteed slot (QOSGrpBilling); the other 7 baselines wait on `normal` priority.
- 2026-11-22 17:21 — Countdown Uniform (93839, `normal`) pre-empted after 10 min by another group's guaranteed job; it requeued itself (restarts from step 0 when it gets a card). Countdown AC (93836) and SEC (93840) unaffected.
- 2026-11-22 17:30 — **DECISION (you): "relaunch the jobs from normal partition to guaranteed".** Done in place with
  `scontrol update JobId=<id> QOS=guaranteed-raul` (job IDs and order kept) for the 7 pending baselines: 93839 countdown
  uniform, 93842–93844 zebra uniform/sec/pcl, 93845–93847 arc uniform/sec/pcl. SLURM refuses a QoS change on a *running*
  job, so Countdown SEC (93840) and PCL (93841) stay on `normal` while they run (they occupy otherwise idle cards and may
  finish in ~12 h); if either is pre-empted it requeues and I move it to guaranteed then. Consequence: `guaranteed-raul`
  is one card for the whole `raul` account, so the 10 queued runs execute strictly one after another, ≈ 12 h (countdown,
  zebra) to 20 h (arc) each → roughly 6–7 days, and the slot is unavailable to other guaranteed work of the account meanwhile.
- 2026-11-22 17:33 — Countdown SEC (93840) was pre-empted as well (after 18 min) and requeued → moved to `guaranteed-raul` while pending. Only Countdown PCL (93841) still runs on `normal`; it will be moved the same way if it requeues.
- 2026-11-22 17:40 — **DECISION (you): move Countdown PCL to guaranteed too.** Cancelled 93841 (`normal`, 9 min in, no checkpoint), its partial TensorBoard dir moved to repro/logs/stale_tb_countdown_pcl_job93841, resubmitted as **93881** on `guaranteed-raul`. All 12 remaining runs are now on the guaranteed slot: running 93836 countdown AC; queued 93837 zebra AC, 93838 arc AC, 93839/93840/93881 countdown uniform/sec/pcl, 93842–93844 zebra, 93845–93847 arc (strictly sequential, ≈ 1 week).
- 2026-11-22 18:00 — **DECISION (you): "run the greedy eval on a guaranteed card".** Submitted **93902** `ac-greedy-eval`
  (repro/greedy_eval.sbatch, guaranteed-raul, pinned to titan-1 where the MATH checkpoints live). It runs the repo's own
  validation (same prompt, verifier, vLLM; `trainer.val_only=true`) with **greedy decoding** (`val_kwargs.do_sample=false`)
  on math_test.parquet for the untrained Qwen2.5-3B and — since they sit on the same node and cost ~5 min each — the
  step-100 checkpoints of all four MATH runs (FSDP → HF via `verl.model_merger`). Results: repro/logs/ac-greedy-eval-93902.out
  (summary at the end), per-model logs repro/logs/greedy_math_<model>.log, TensorBoard under tensorboard_log/actor_curator_eval/.
  To make it run right after Countdown AC (the guaranteed slot is FIFO by age), the 11 pending chain jobs were cancelled
  and resubmitted *behind* it (no training lost — none had started): 93903 zebra AC, 93904 arc AC, 93905/93906/93907
  countdown uniform/sec/pcl, 93908–93910 zebra uniform/sec/pcl, 93911–93913 arc uniform/sec/pcl. Step-0-only TensorBoard
  dirs of the pre-empted countdown uniform/sec runs moved to repro/logs/stale_tb_*. Fixed a dry-run display bug in
  repro/run.sh (`diff | grep` under pipefail aborted before printing the override list; real runs unaffected).
- 2026-11-22 18:35 — Countdown AC (93836): 350 s/step (~10 h total). Validation (easy/medium/hard/very_hard, %): step 0 2.0/1.5/0/0 · step 10 1.0/3.0/0.5/0.5; training reward mean step 5 = 0.24 % (≈5 correct of 2048 responses). Curator still dormant until step 25.
- 2026-11-25 14:30 — **Status after 3 days on the guaranteed slot.** Finished (100 steps each): Countdown AC 93836 (4.3 h),
  Zebra AC 93903 (8.2 h), ARC AC 93904 (16.4 h), Countdown Uniform 93905 (5.7 h), Countdown PCL 93907 (4.1 h),
  Zebra Uniform 93908 (7.7 h), Zebra SEC 93909 (7.4 h), Zebra PCL 93910 (7.3 h). Running: ARC Uniform 93911 (titan-2, step ~45).
  **Countdown SEC 93906 FAILED at launch** (Ray: "The current node timed out during startup … GCS overloaded"): it started
  2 s after 93905 ended on the same node while that job's Ray cluster was still shutting down → transient. Resubmitted as
  **96518**; repro/run.sh now retries once if the launch dies within 10 min (queued jobs carry the script version they were
  submitted with, so only new submissions get the retry).
  **Greedy eval 93902 has not run**: it is pinned to titan-1 (where the MATH checkpoints are) and titan-1 is entirely held
  by another group's guaranteed job (8 GPUs + all memory, time limit → 2026-11-27 00:16). Guaranteed cannot pre-empt
  guaranteed, so 93902 waits for that job to end; SLURM meanwhile runs my later jobs around it (that is why the chain
  progressed). DECISION: added an unpinned **base-model-only** greedy eval (**96517**, repro/greedy_eval_base.sbatch) so the
  key number (untrained Qwen2.5-3B greedy MATH500 vs. paper π_ref 61.8) comes after ARC Uniform instead of on Nov 27;
  93902 stays queued for the four trained checkpoints. ARC SEC/PCL requeued behind these two: **96519 / 96520**.
  Harmless noise in logs: "RuntimeError: DataLoader worker … killed by signal: Killed" appears only during Ray teardown
  after the final checkpoint of COMPLETED runs; "Error ValueError …" lines are model-generated text in validation samples.

  **Results so far** (peak within 100 steps, %; "hard" = `-very_hard-test`, see B8; mean4 = mean of the 4 levels):

  | | Uniform | SEC | PCL | AC | paper U / SEC / PCL / AC |
  |---|---|---|---|---|---|
  | Countdown mean4 | 36.9 | (96518 queued) | 30.1 | 32.3 | 44.7 / 58.9 / 57.2 / 62.1 |
  | Countdown very_hard | 10.5 | – | 10.0 | 15.0 | 41.0 / 51.5 / 48.0 / 58.0 |
  | Countdown hard-test (3rd level) | 22.0 | – | 19.0 | 20.5 | – |
  | Zebra mean4 | 44.5 | 43.6 | 43.6 | 45.0 | 35.1 / 36.0 / 34.1 / 37.6 |
  | Zebra very_hard | 29.0 | 26.5 | 24.0 | 26.0 | 30.5 / 27.5 / 26.0 / 34.5 |
  | ARC-1D mean4 | (step 40: 9.4) | queued | queued | 37.3 | 26.7 / 27.9 / 26.4 / 36.4 |
  | ARC-1D very_hard | (step 40: 3.0) | queued | queued | 25.0 | 19.5 / 18.5 / 18.5 / 31.0 |
  | MATH500 | 64.2 | 64.4 | 65.4 | 64.2 | 83.0 / 81.0 / 79.8 / 81.0 |
  | AIME24 | 10.0 | 10.0 | 6.7 | 16.7 | 23.3 / 20.0 / 23.3 / 30.0 |

  Reading: ARC AC matches the paper (37.3 vs 36.4); Zebra is *above* the paper for every method but shows no AC advantage
  (all within 1.4 points, one seed); Countdown is far below the paper for every method (AC 32 vs 62) and Uniform beats AC
  there (one seed; all Countdown curves were still rising at step 100). Base-model step-0 accuracies match the paper's
  π_ref = 0 for Countdown/Zebra/ARC (ours 0.9 / 8.0 / 1.4 % mean4).

- 2026-11-26 10:30 — **Status.** Finished: ARC Uniform 93911 (15.1 h; peak mean4 36.9, very_hard 23.5 — same as ARC AC 37.3/25.0,
  paper Uniform 26.7/19.5), Countdown SEC 96518 (4.4 h; peak mean4 26.8, very_hard 8.5 — lowest of the four Countdown methods,
  paper 58.9/51.5), **greedy eval of the untrained model 96517** (4.7 min): MATH500 26.8 %, AMC 15.7 %, AIME24 0 % with greedy
  decoding → the paper's π_ref 61.8 is not a temperature effect (details in A7). Running: ARC SEC 96519 on titan-2 since 01:51,
  step 40 at 7.6 h (mean4 4.9 %; ARC Uniform was 9.4 % at step 40 and ended at 36.9), ETA ≈ 21:00 today. Queued on the one
  guaranteed card (FIFO by age): 93902 greedy eval of the four MATH step-100 checkpoints, pinned to titan-1, SLURM-estimated
  start 2026-11-27 00:16 (when the other group's job 92747 ends); then 96520 ARC PCL (~16–19 h → done ≈ Nov 27 evening / Nov 28
  morning). The card will probably sit idle ≈3 h between ARC SEC and 93902 (backfill cannot start the 2.5-day ARC PCL without
  delaying the older 93902). DECISION: left as is — copying the checkpoints off titan-1 would itself need a job on titan-1.
  13 of 16 cells done; remaining: ARC SEC (running), ARC PCL (queued), plus the checkpoint greedy eval.

  **Results so far** (peak within 100 steps, %; "hard" = `-very_hard-test`, see B8; mean4 = mean of the 4 levels):

  | | Uniform | SEC | PCL | AC | paper U / SEC / PCL / AC |
  |---|---|---|---|---|---|
  | Countdown mean4 | 36.9 | 26.8 | 30.1 | 32.3 | 44.7 / 58.9 / 57.2 / 62.1 |
  | Countdown very_hard | 10.5 | 8.5 | 10.0 | 15.0 | 41.0 / 51.5 / 48.0 / 58.0 |
  | Zebra mean4 | 44.5 | 43.6 | 43.6 | 45.0 | 35.1 / 36.0 / 34.1 / 37.6 |
  | Zebra very_hard | 29.0 | 26.5 | 24.0 | 26.0 | 30.5 / 27.5 / 26.0 / 34.5 |
  | ARC-1D mean4 | 36.9 | (step 40: 4.9) | queued | 37.3 | 26.7 / 27.9 / 26.4 / 36.4 |
  | ARC-1D very_hard | 23.5 | (step 40: 1.5) | queued | 25.0 | 19.5 / 18.5 / 18.5 / 31.0 |
  | MATH500 | 64.2 | 64.4 | 65.4 | 64.2 | 83.0 / 81.0 / 79.8 / 81.0 |
  | AIME24 | 10.0 | 10.0 | 6.7 | 16.7 | 23.3 / 20.0 / 23.3 / 30.0 |
  | MATH500 untrained, greedy / sampled | 26.8 / 16.8 | | | | π_ref 61.8 |

  Reading (one seed each): AC ≈ Uniform on ARC and Zebra (paper: AC +10 and +2.5 over Uniform); our Uniform baselines are much
  stronger than the paper's on ARC/Zebra, and every Countdown method is far below the paper. The curator shows no measurable
  gain over random selection in this reproduction so far.
