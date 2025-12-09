import datetime
import logging
import os
import numpy as np
import torch
import orjson
import algorithms
import random
from utils import make_valid_json, record
import time
import env_causalLM as environment
# import env_maskedLM as environment 

os.environ["TOKENIZERS_PARALLELISM"] = "true"

seeds = [0, 1, 42, 43, 100]
tasks = [
    "mnli", "qqp", "sst2", "mrpc", "cola", "qnli", 
    # "rte"
    ]

default_alg_opts = {"lambda": 20, "mu": 10, "verbose": 2, "maxFEs": 5000, "test_times": 10}
eff_dim = 500
tau = 1.0 / (2 * eff_dim / default_alg_opts["mu"]) ** 0.5
alg_configs = [
    # ("bbt", algorithms.bbt, {"sigma0": 1.0, "eff_dim": 500}),
    ("SAES_id", algorithms.SAES_id_e, {"sigma0": 1e0, "eff_dim": eff_dim}),
]
model_name = "facebook/opt-6.7b"
# model_name = "Roberta-large"

coefs = [
    # 0.01, 
    0.1]

result_dir = f"./results/{model_name}"
for coef in coefs:
    for task in tasks:
        for seed in seeds:
            for alg_name, alg_func, alg_settings in alg_configs:
                result_path = os.path.join(result_dir, task, alg_name)
                os.makedirs(result_path, exist_ok=True)
                result_file = os.path.join(result_path, f"coef{coef}-seed{seed}-{datetime.datetime.now().strftime('%H-%M-%S')}.json")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            # ---- Logging & dirs ----
            log_dir = f"logs/{model_name}/{task}"
            os.makedirs(log_dir, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                filename=os.path.join(log_dir, f"{datetime.datetime.now().strftime('%d-%H-%M')}-{alg_name}-{seed}.log")
            )
            logger = logging.getLogger(__name__)
            logger.info(f"Starting run for task: {task}, seed: {seed}, algorithm: {alg_name}")
            opt = environment.EnvOptions(n=10, task=task, coef=coef, mdl_name=model_name)
            print(f"EnvOptions: {opt} for algorithm: {alg_name}, seed: {seed}")
            env = environment.Environment(opt)
            env.prepare_data()
            ds = {
                "train": env.get_fewshot_dataloader(seed=seed),
                "val": env.get_fewshot_dataloader(seed=seed),
                "test": env.get_fulldataloader(),
            }

            x0 = list(range(1000, 1000 + env.n))
            x0 = env.ebd(torch.tensor(x0).cuda()).cpu().reshape(-1)
            x0 = x0.detach()

            alg_opts = dict(default_alg_opts)
            alg_opts.update(alg_settings)
            logger.info(f"[tuned hyperparameters] task:{task} sigma0:{alg_opts['sigma0']}, coefficient:{opt.coef}, seed:{seed} alg:{alg_name}")
            logger.info(f"Starting optimization for {alg_name} with maxFEs={alg_opts.get('maxFEs')}")

            start_time = time.perf_counter()
            out = alg_func(env, ds, alg_opts, x0)
            end_time = time.perf_counter()
            wall_clock_time_seconds = end_time - start_time
            logger.info(f"Optimization finished. Total Wall-Clock Time: {wall_clock_time_seconds:.2f} seconds")
            out["wall_clock_time_seconds"] = wall_clock_time_seconds
            out = make_valid_json(out)

            with open(result_file, "wb") as f:
                f.write(orjson.dumps(out, option=orjson.OPT_SERIALIZE_NUMPY))

