import datetime
import logging
import os
import numpy as np
import torch
import orjson
import  algorithms 
import random
import math
from utils import make_valid_json
import argparse
import time
os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser=argparse.ArgumentParser()
parser.add_argument("--coef", type=str, default="(10,0.1)",)
parser.add_argument("--sigma0",type=float,default=1e-2)
parser.add_argument("--task", type=str,default="sst2")
parser.add_argument("--seed", type=int, default=42, choices=[0, 1, 42, 43, 100], help="Random seed")
parser.add_argument("--model", type=str, default="facebook/opt-6.7b", help="Pretrained model name or local Models folder name")
parser.add_argument("--eff_dim", type=int, default=500, help="effective dimension (id) used by SA-ES-id")
parser.add_argument("--maxFEs", type=int, default=5000, help="maximum function evaluations for each run")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if "Roberta" in args.model:
    import env_maskedLM as environment
elif "opt" in args.model:
    import env_causalLM as environment
else:
    import env_seq2seqLM as environment

s = args.coef.strip()
if s.startswith("(") and s.endswith(")"):
    a, b = s[1:-1].split(",")
    coef = (float(a), float(b))
else:
    coef = float(s)

result_dir = f"./results/{args.model}"
opt = environment.EnvOptions(n=10, task=args.task, coef=coef, mdl_name=args.model)

log_dir = f"logs/{args.model}/{args.task}"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(log_dir, f"{datetime.datetime.now().strftime('%d-%H-%M')}.log")
)
logger = logging.getLogger(__name__)
print(opt)
env = environment.Environment(opt)

env.prepare_data()
ds = {
    "train": env.get_fewshot_dataloader(),
    "val": env.get_fewshot_dataloader(),
    "test": env.get_fulldataloader(),
}

x0 = list(range(1000, 1000 + env.n))
if torch.cuda.is_available():
    x0 = env.ebd(torch.tensor(x0).cuda()).cpu().reshape(-1)
else:
    x0 = env.ebd(torch.tensor(x0)).cpu().reshape(-1)
x0 = x0.detach()

default_alg_opts = {"lambda": 20, "mu": 10, "verbose": 2, "maxFEs": args.maxFEs, "test_times": 10}

# Prepare experiments: first algorithm (SAES_id) single tau; second algorithm (SAES_id_e) three taus
id_val = args.eff_dim
lam = 4 + int(math.ceil(3 * math.log(id_val)))
mu = int(math.ceil(lam / 2))
tau_default = 1.0 / (2 * id_val) ** 0.5
tau_mu = 1.0 / (2 * id_val / mu) ** 0.5
tau_lam = 1.0 / (2 * id_val / lam) ** 0.5

experiments = [
    # (algorithms.SAES_id, {"sigma0": args.sigma0, "eff_dim": id_val, "tau": tau_default}),
    # (algorithms.SAES_id_e, {"sigma0": args.sigma0, "eff_dim": id_val, "tau": tau_default}),
    # (algorithms.SAES_id_e, {"sigma0": args.sigma0, "eff_dim": id_val, "tau": tau_lam}),
    (algorithms.SAES_id_e, {"sigma0": args.sigma0, "eff_dim": id_val, "tau": tau_mu}),
]
for alg_func, settings in experiments:
    alg_opts = dict(default_alg_opts)
    alg_opts.update(settings)
    # include tau in folder name for clarity
    tau_tag = f"_tau_{alg_opts.get('tau'):.6g}" if "tau" in alg_opts else ""
    result_path = os.path.join(result_dir, env.task, "different_tau", alg_func.__name__ + tau_tag)
    os.makedirs(result_path, exist_ok=True)
    result_file = os.path.join(result_path, f"{env.task}-{datetime.datetime.now().strftime('%H-%M')}.json")

    logger.info(f"[tuned hyperparameters] task:{opt.task} sigma0:{alg_opts.get('sigma0')}, coefficient:{opt.coef}, tau:{alg_opts.get('tau')}")
    logger.info(f"Starting optimization for {alg_func.__name__} with maxFEs={alg_opts.get('maxFEs')}")
    start_time = time.perf_counter()

    out = alg_func(env, ds, alg_opts, x0)

    end_time = time.perf_counter()
    wall_clock_time_seconds = end_time - start_time
    logger.info(f"Optimization finished. Total Wall-Clock Time: {wall_clock_time_seconds:.2f} seconds")
    out["wall_clock_time_seconds"] = wall_clock_time_seconds
    out["alg_opts"] = alg_opts

    out = make_valid_json(out)
    with open(result_file, "wb") as f:
        f.write(orjson.dumps(out, option=orjson.OPT_SERIALIZE_NUMPY))

    # try to log final reported metrics if present
    try:
        i = np.argmax([o["metric"] for o in out["test"][:-2]])
        logger.info(f"[final results] with validation:{out['test'][-1]['metric']} with check points:{out['test'][i]['metric']}")
    except Exception:
        logger.info("[final results] could not parse test metrics from output JSON")