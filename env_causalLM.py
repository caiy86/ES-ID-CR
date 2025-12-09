from functools import partial
from typing import Any
import evaluate
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding,set_seed
import torch
from datasets import load_dataset, load_from_disk, DatasetDict,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

CELoss = torch.nn.CrossEntropyLoss()
UnreducedCELoss = torch.nn.CrossEntropyLoss(reduction="none")
@dataclass
class EnvOptions:

    task: str = "sst2"
    n: int = 5
    mdl_name: str = "facebook/opt-6.7b"
    device: torch.device = torch.device("cuda:0")
    nShots: int = 16
    # truncate the test set
    test_samples: int = 1000
    max_sentence_length: int = 450
    cuda_batch_size: int = 64
    coef: Any=0.0

GLUE_DATASETS = ["mnli", "qqp", "sst2", "mrpc", "cola", "wnli", "qnli", "rte"]

class Environment:
    def info(env):
        return f"size: (train={env.datasets['train'].shape[0]},test={env.datasets['test'].shape[0]})"

    # wrap the dataset with dataloaders, and build the validation set if required
    # use cache whenever possible
    def prepare_data(env):
        try:
            ds = load_from_disk(f"./data_store/prepared/glue/{env.mdl_name}/{env.task}")
        except:
            print("Prepare data and then cache...")
            if not hasattr(env, "raw_data"):
                # load data and preprocessing
                env.raw_data = load_raw_data(env.task)

            ds = prepare_train_test_sets(env)
            ds.save_to_disk(f"./data_store/prepared/glue/{env.mdl_name}/{env.task}")

        if ds["test"].shape[0] > env.test_samples:
            idxs = np.random.choice(ds["test"].shape[0], env.test_samples, False)
            ds["test"] = ds["test"].select(idxs)

        env.datasets = ds

    def get_fulldataloader(env, name="test"):
        samples = env.datasets[name]
        tokenized_data = env.tok(
            samples["templated_sentence"],
            padding=False,
            max_length=env.max_sentence_length,
            truncation=True,
        )
        data = Dataset.from_dict(tokenized_data).add_column("labels", samples["labels"])
        dataloader = env.dataloader_creator(data, shuffle=True)
        return dataloader

    def get_fewshot_dataloader(env, seed: int = None):
        samples = env.datasets["train"]
        randidxs = []
        for l in np.unique(samples["labels"]):
            candidates = np.nonzero(samples["labels"] == l)[0]
            idxs = np.random.choice(candidates, env.nShots, replace=False)
            randidxs.extend(idxs.tolist())
        fewshot = samples.select(randidxs)

        tokenized_data = env.tok(
                    fewshot["templated_sentence"],
                    padding=False,
                    max_length=env.max_sentence_length,
                    truncation=True,
                )
        data = Dataset.from_dict(tokenized_data).add_column("labels", fewshot["labels"])       
        dataloader = env.dataloader_creator(data, shuffle=False)

        return dataloader
     
    # @torch.no_grad()
    # def forward(env, dataloader, X):
    #     X = X.reshape(1, env.n, env.d).to(env.device)
    #     ref = []
    #     logits = []
    #     prob = []  
    #     for batch in dataloader: 
    #         b = batch["input_ids"].shape[0]
    #         input_ids = batch["input_ids"].to(env.device)
    #         labels = batch["labels"].to(env.device)

    #         # b * l * d
    #         inputs_embeds = env.ebd(input_ids)
    #         inputs_embeds_new = torch.cat(
    #         [inputs_embeds[:, 0, :].unsqueeze(1),  X.repeat(b, 1, 1), inputs_embeds[:, 1:, :]], dim=1
    #     )
    #         attention_mask = torch.cat([batch["attention_mask"][:, 0].unsqueeze(1),torch.ones(b, env.n), batch["attention_mask"][:,1:]], dim=1).to(env.device)
            
    #         # model inference
    #         out = env.mdl(inputs_embeds=inputs_embeds_new, attention_mask=attention_mask)["logits"]

    #         seq_lengths = attention_mask.sum(dim=1).long()            
    #         batch_indices = torch.arange(b, device=env.device)
    #         out = out[batch_indices, seq_lengths - 1, :]

    #         all_prob = out.softmax(dim=1)  # probabilities of vocabularies: b * V
    #         verbalizers_prob = all_prob[..., env.tokenized_label]  # probabilities of verbalizers: b * C

    #         # the probability that the predictions gives the desired verbalizers
    #         prob.append(verbalizers_prob.sum(dim=1))

    #         ref.append(labels)

    #         out = out[..., env.tokenized_label]
    #         assert not out.isnan().any()
    #         logits.append(out)

    #     logits = torch.cat(logits)
    #     ref = torch.cat(ref)
    #     prob = torch.cat(prob)

    #     torch.cuda.empty_cache()

    #     return logits, prob, ref
    @torch.no_grad()
    def forward(env, dataloader, X):
        X = X.reshape(1, env.n, env.d).to(env.device)
        ref = []
        logits = []
        prob = []  
        vocab_size = env.ebd.weight.shape[0]
        for batch in dataloader: 
            b = batch["input_ids"].shape[0]
            input_ids = batch["input_ids"].to(env.device)
            labels = batch["labels"].to(env.device)
            attention_mask_batch = batch["attention_mask"].to(env.device)
            if input_ids.max() >= vocab_size:
                print(f"Found input_id {input_ids.max()} >= vocab size {vocab_size}")
                raise ValueError("Input ID out of bounds!")
            inputs_embeds = env.ebd(input_ids)
            
            embeds_bos = inputs_embeds[:, 0, :].unsqueeze(1) 
            embeds_prompt = X.repeat(b, 1, 1) 
            embeds_text = inputs_embeds[:, 1:, :] 

            inputs_embeds_new = torch.cat([embeds_bos, embeds_prompt, embeds_text], dim=1)

            mask_bos = attention_mask_batch[:, 0].unsqueeze(1)
            mask_prompt = torch.ones(b, env.n, device=env.device, dtype=torch.long)
            mask_text = attention_mask_batch[:, 1:]

            attention_mask = torch.cat([mask_bos, mask_prompt, mask_text], dim=1)
            if inputs_embeds_new.shape[1] > 2048:
                 print(f"Error: Sequence length {inputs_embeds_new.shape[1]} exceeds model limit 2048")
                 raise ValueError("Sequence too long for OPT model")
            out = env.mdl(inputs_embeds=inputs_embeds_new, attention_mask=attention_mask)["logits"]

            seq_lengths = attention_mask.sum(dim=1).long()
            batch_indices = torch.arange(b, device=env.device)
            out = out[batch_indices, seq_lengths-1, :]

            all_prob = out.softmax(dim=1)  
            verbalizers_prob = all_prob[..., env.tokenized_label]  
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
        return loss.item(), metric
    
    def __init__(env, opts):  # initialize fields provided by options
        for k, v in vars(opts).items():
            setattr(env, k, v)
        # load LM
        mdl_file = f"../Models/{opts.mdl_name}"
        env.tok = AutoTokenizer.from_pretrained(mdl_file)
        env.tok.pad_token = env.tok.eos_token
        env.mdl = AutoModelForCausalLM.from_pretrained(mdl_file, torch_dtype=torch.bfloat16).to(env.device)
        env.mdl.eval()
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
    "mnli": {"yes": 0, "maybe": 1, "no": 2},
    "qqp": {"no": 0, "yes": 1},
    "sst2": {"terrible": 0, "great": 1},
    "mrpc": {"no": 0, "yes": 1},
    "cola": {"no": 0, "yes": 1},
    "wnli": {"no": 0, "yes": 1},
    "qnli": {"yes": 0, "no": 1},
    "rte": {"yes": 0, "no": 1},
}

TEMPLATE_CONFIG = {
    "sst2": "What is the sentiment of this movie review? <s1> This movie was ",
    "cola": "<s1> This is yes or no?",
    "mnli": "<s1> Does this mean that '<s2>' is true? yes, maybe or no?",
    "qnli": "Dose <s2> answers the question '<s1>'? yes or no?",
    "rte": "<s1> Does this mean that '<s2>' is true? yes or no?",
    "mrpc": "Do <s2> and <s1> have the same meaning? yes or no?",#!
    "qqp": "Do <s2> and <s1> have the same meaning? yes or no?",  # !
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
    
    key1, key2 = TASK_TO_KEYS[env.task]
    def add_template(s):
        if key2:
            s1 = s[key1]
            s2 = s[key2]
            s["templated_sentence"] = template.replace("<s1>", s1).replace("<s2>", s2)
        else:
            s1 = s[key1]
            s["templated_sentence"] = template.replace("<s1>", s1)
        return s
    useful_columns = ["templated_sentence", "labels"]
    for k, v in ds.items():
        # add template and truncate
        tmp = v.map(add_template, load_from_cache_file=False, desc="add template:" + k)
        tmp = tmp.rename_column("label", "labels")
        # remove unnecessary columns
        ds[k] = tmp.remove_columns([n for n in tmp.column_names if not n in useful_columns])
    return DatasetDict(ds)