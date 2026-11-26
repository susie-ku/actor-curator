# repro/ — end-to-end reproduction of Table 1 (Qwen2.5-3B) on the titan SLURM cluster

Nothing under `verl/` or `recipe/` is modified. See `../PROBLEMS.md` for every decision/deviation.

* `install_env.sh`  – builds the `verl` conda env (torch 2.7.1+cu128, vLLM 0.10.0, flash-attn 2.8.0.post2) for Blackwell GPUs.
* `gpu_check.sh/.py` – 5-minute sanity job: torch sm_120, flash-attn fwd/bwd, vLLM generation.
* `run.sh`          – SLURM job for one cell: `BENCH={countdown,zebra,arc,math}` × `METHOD={ac,uniform,sec,pcl}`.
                      Copies the recipe script to `repro/tmp/`, patches shell variables with sed, appends Hydra overrides.
* `submit.sh`       – `repro/submit.sh countdown ac` (add `SMOKE=1` for the 3-step pipeline check).
* `scorer_check.py` – unit check of the patched reward extractor (A8) through the `sec` reward manager: `envs/verl/bin/python repro/scorer_check.py`.
* `collect_results.py` – reads `tensorboard_log/actor_curator_repro/*` and prints peak val accuracy ≤ step 100 per data source.
* `logs/`           – SLURM stdout/stderr per job.  Checkpoints go to node-local `/scratch/$USER/actor-curator-ckpt/`.
