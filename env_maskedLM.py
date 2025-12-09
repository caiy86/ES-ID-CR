from collections import defaultdict
import copy
from functools import partial
import itertools
import logging
import math
import os
from typing import Any
import evaluate
import numpy as np
from os.path import isfile
from imblearn.metrics import geometric_mean_score
from dataclasses import dataclass
import orjson
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding,set_seed
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, RobertaForMaskedLM, AutoModelForSeq2SeqLM, GenerationConfig

CELoss = torch.nn.CrossEntropyLoss()
UnreducedCELoss = torch.nn.CrossEntropyLoss(reduction="none")
@dataclass
class EnvOptions:

    task: str = "sst2"
    n: int = 5
    mdl_name: str = "Roberta-large"
    device: torch.device = torch.device("cuda:0")
    # number of each shot in a minibatch
    nShots: int = 16
    # truncate the test set
    test_samples: int = 1000
    max_sentence_length: int = 450
    # batch size of cuda operations
    cuda_batch_size: int = 64
    coef: Any=0.0


DOMAIN_DATASET = ["CI", "SE", "RCT", "HP"]
GLUE_DATASETS = ["mnli", "qqp", "sst2", "mrpc", "cola", "wnli", "qnli", "rte"]

class Environment:
    def info(env):
        return f"size: (train={env.datasets['train'].shape[0]},test={env.datasets['test'].shape[0]})"

    # wrap the dataset with dataloaders, and build the validation set if required
    # use cache whenever possible
    def prepare_data(env):
        # if possible, use saved data sets
        try:
            ds = load_from_disk(f"./data_store/prepared/glue/{env.mdl_name}/{env.task}")
        except:
            print("Prepare data and then cache...")
            if not hasattr(env, "raw_data"):
                # load data and preprocessing
                env.raw_data = load_raw_data(env.task)

            ds = prepare_train_test_sets(env)
            ds.save_to_disk(f"./data_store/prepared/glue/{env.mdl_name}/{env.task}")

        # truncate the test set
        if ds["test"].shape[0] > env.test_samples:
            idxs = np.random.choice(ds["test"].shape[0], env.test_samples, False)
            ds["test"] = ds["test"].select(idxs)

        env.datasets = ds

    def get_fulldataloader(env, name="test"):
        samples = env.datasets[name]
        dataloader = env.dataloader_creator(samples, shuffle=True)
        return dataloader

    def get_fewshot_dataloader(env, seed: int = None):
        samples = env.datasets["train"]

        # 按每个标签随机挑 env.nShots 个样本
        randidxs = []
        for l in np.unique(samples["labels"]):
            candidates = np.nonzero(samples["labels"] == l)[0]
            idxs = np.random.choice(candidates, env.nShots, replace=False)
            randidxs.extend(idxs.tolist())

        # 构造 DataLoader 并返回
        fewshot = samples.select(randidxs)
        dataloader = env.dataloader_creator(fewshot, shuffle=False)
        return dataloader
     
    @torch.no_grad()
    def forward(env, dataloader, X):
        X = X[0 : env.n * env.d].reshape(1, env.n, env.d).cuda()  # prompt
        ref = []
        logits = []
        prob = []  
        for batch in dataloader:
            b = batch["input_ids"].shape[0]
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            # b * l * d
            inputs_embeds = env.ebd(input_ids)
            inputs_embeds_new = torch.cat(
                [inputs_embeds[:, 0, :].unsqueeze(1), X.repeat(b, 1, 1), inputs_embeds[:, 1:, :]], dim=1
            ).cuda()
            attention_mask = torch.cat([torch.ones(b, env.n), batch["attention_mask"]], dim=1).cuda()
            
            # model inference
            out = env.mdl(inputs_embeds=inputs_embeds_new, attention_mask=attention_mask)["logits"]

            mask_loc = torch.cat(
                [
                    torch.zeros(b, env.n, dtype=torch.bool, device=X.device),
                    input_ids == env.tok.mask_token_id,
                ],
                dim=1,
            )
            # focus on masks
            out = out[mask_loc]

            all_prob = out.softmax(dim=1)  # probabilities of vocabularies: b * V
            verbalizers_prob = all_prob[..., env.tokenized_label]  # probabilities of verbalizers: b * C

            # the probability that the predictions gives the desired verbalizers
            prob.append(verbalizers_prob.sum(dim=1))

            ref.append(labels)

            out = out[..., env.tokenized_label]
            assert not out.isnan().any()
            logits.append(out)

        logits = torch.cat(logits)
        ref = torch.cat(ref)
        prob = torch.cat(prob)

        torch.cuda.empty_cache()

        return logits, prob, ref

    @torch.no_grad()
    def objective2(env, dataloader, X):
        logits, probs, ref = env.forward(dataloader, X)

        pred = logits.argmax(dim=-1)

        metric = env.metric.compute(predictions=pred, references=ref)
        metric = [*metric.values()][0]
        loss1 = CELoss(logits, ref).item()
        loss2 = -probs.log().mean().item()

        return loss1, loss2, metric

    @torch.no_grad()
    def objective(env, dataloader, X):
        logits, probs, ref = env.forward(dataloader, X)

        pred = logits.argmax(dim=-1)

        metric = env.metric.compute(predictions=pred, references=ref)
        metric = [*metric.values()][0]

        loss = CELoss(logits, ref)
        loss -= env.coef * probs.log().mean()
        return loss.item(), metric
    
    def __init__(env, opts):  # initialize fields provided by options
        for k, v in vars(opts).items():
            setattr(env, k, v)
        # load LM
        mdl_file = f"../Models/{opts.mdl_name}"
        env.tok = AutoTokenizer.from_pretrained(mdl_file)
        env.mdl = RobertaForMaskedLM.from_pretrained(mdl_file).eval().cuda()     

        env.ebd = env.mdl.get_input_embeddings()
        env.d = env.ebd.embedding_dim

        # load metric
        env.metric = evaluate.load("metrics/" + get_metric_name(opts.task))

        # used for indexing the LLM output
        env.tokenized_label = torch.tensor(
            [env.tok.encode(k, add_special_tokens=False)[0] for k, v in LABEL2ID_CONFIG[opts.task].items()],
            device=env.mdl.device,
        )
        env.nClasses = len(env.tokenized_label)

        env.dataloader_creator = partial(
            DataLoader,
            batch_size=env.cuda_batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer=env.tok),
            pin_memory=True,
        )
     
def get_metric_name(name):
    if name == "cola":
        m = "matthews_correlation"
    elif name in ["mnli", "sst2", "wnli", "rte", "qnli", "MR", "CR"]:
        m = "accuracy"
    else:
        m = "f1"
    return m


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

LABEL2ID_CONFIG = {
    "mnli": {" yes": 0, " maybe": 1, " no": 2},
    "qqp": {" no": 0, " yes": 1},
    "sst2": {" terrible": 0, " great": 1},
    "mrpc": {" no": 0, " yes": 1},
    "cola": { " no": 0," yes": 1},
    "wnli": {" no": 0, " yes": 1},
    "qnli": {" yes": 0, " no": 1},
    "rte": {" yes": 0, " no": 1},
}

TEMPLATE_CONFIG = {
    "sst2": "<s1> It was [MASK] .",
    "cola": "<s1> This is [MASK] .",
    "mnli": "<s1>? [MASK] , <s2>",
    "qnli": "<s1>? [MASK] , <s2>",
    "rte": "<s1>? [MASK] , <s2>",
    "mrpc": "<s1> [MASK] , <s2>",
    "qqp": "<s1> [MASK] , <s2>",  
}

def load_raw_data(task):
    try:
        ds = load_from_disk("../DataSet/glue/" + task)
    except:
        ds = load_dataset("glue", task)
        ds.save_to_disk("../DataSet/glue" + task)
    return ds

def prepare_train_test_sets(env):
    ds = {}
    ds["train"] = env.raw_data["train"]
    ds["test"] = env.raw_data["validation_matched"] if env.task == "mnli" else env.raw_data["validation"]

    template = TEMPLATE_CONFIG[env.task]
    template = template.replace("[MASK]", env.tok.mask_token)

    # truncate the sentence to length l
    trunc = lambda s, l: env.tok.convert_tokens_to_string(env.tok.tokenize(s)[:l])

    key1, key2 = TASK_TO_KEYS[env.task]
    def add_template(s):
        if key2:
            s1 = trunc(s[key1], 200)
            s2 = trunc(s[key2], 200)
            s["templated_sentence"] = template.replace("<s1>", s1).replace("<s2>", s2)
        else:
            s1 = trunc(s[key1], 400)
            s["templated_sentence"] = template.replace("<s1>", s1)
        return s
    useful_columns = ["input_ids", "attention_mask", "labels"]
    for k, v in ds.items():
        # add template and truncate
        tmp = v.map(add_template, load_from_cache_file=False, desc="add template:" + k)
        # tokenize the input sentences (which involve the template)
        tmp = tmp.map(
            lambda b: env.tok(
                b["templated_sentence"],
                padding=False,
                max_length=env.max_sentence_length,
                truncation=True,
            ),
            load_from_cache_file=False,
            desc="tokenize:" + k,
        )
        tmp = tmp.rename_column("label", "labels")
        # remove unnecessary columns
        ds[k] = tmp.remove_columns([n for n in tmp.column_names if not n in useful_columns])

    return DatasetDict(ds)

def add_init_prompt(examples,max_len):
    for i in range(len(examples['input_ids'])):
        if len(examples["input_ids"][i])>max_len:
            continue
        init_prompt= list(range(1000,1000+max_len-len(examples["input_ids"][i])))
        examples["input_ids"][i]=[examples["input_ids"][i][0]]+init_prompt+examples["input_ids"][i][1:]
        examples["attention_mask"][i]=[1]*len(init_prompt)+examples["attention_mask"][i]
    return examples