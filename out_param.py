import numpy as np
import math, os, sys, types, time, gc
import torch
import torch.nn as nn
from src.utils import TOKENIZER
from src.binidx import MMapIndexedDataset
from src.utils import Dataset
import matplotlib.ticker as ticker
import pandas as pd
import statistics
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast,AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import AdamW

from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from src.spikingjelly.clock_driven import functional
try:
    print("huh")
    #os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    #os.environ["CUDA_HOME"] = "/home/pasindu/miniconda3/envs/spikegpt/lib/python3.10/site-packages/torch/cuda"
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()


######################################################
# Load Test Database for Inference
######################################################

epoch_length_fixed = 100
device = "cuda:0"



tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

tokenizer.pad_token = "<|padding|>"
def tokenize(batch):
    return tokenizer(batch["sentence"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset = load_dataset("nyu-mll/glue", "sst2", keep_in_memory=True, cache_dir="/share/datasets")
#dataset = datasets.load_from_disk('sst-2')
#dataset.save_to_disk('/share/datasets/subj')
train_dataset = dataset["train"]
test_dataset = dataset["validation"]
eval_dataset = dataset["test"]
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=16)
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=64)
eval_dataset = eval_dataset.map(tokenize, batched=True, batch_size=64)
def collate_fn(examples):
    examples = tokenizer.pad(
            examples,
            padding=True,
            max_length=None,
        )
    new_batch_data = []
    new_batch_label = []

    for i in range(len(examples['input_ids'])):
        new_batch_data.append(torch.tensor(examples['input_ids'][i]))
        new_batch_label.append(torch.tensor(examples['label'][i], dtype=torch.long))
    data = torch.stack(new_batch_data, dim=0)
    label = torch.stack(new_batch_label, dim=0)
    return data, label
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=12, shuffle=False, collate_fn=collate_fn)




#def gen_pbar():         #For loading in test data
#    epoch_length_fixed = 100
#    datafile_test = 'wikitext-103.test_text_document'

#    test_dataset = Dataset(MMapIndexedDataset(datafile_test), 100, epoch_length_fixed)
#    print(test_dataset)
#    loader = DataLoader(test_dataset, shuffle=False, batch_size=4)

#    pbar = tqdm(enumerate(loader), total=len(
#                loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
#    return pbar
args.RUN_DEVICE = "cuda" # 'cuda' // 'cpu' (already fast)
args.FLOAT_MODE = "fp32" # fp16 (good for GPU, does not work for CPU) // fp32 (good for CPU) // bf16 (less accurate, but works for CPU)

# if args.RUN_DEVICE == "cuda":
#     os.environ["RWKV_RUN_BACKEND"] = 'nvfuser' # !!!BUGGY!!! wrong output
os.environ["RWKV_JIT_ON"] = '1' # '1' or '0'. very useful for GPU/CPU fp32, but might be harmful for GPU fp16. please benchmark !!!

#For BookCorpus Pre-trained model
# TOKEN_MODE = "char"
# WORD_NAME = "vocab_book"
# UNKNOWN_CHAR = ' '
# vocab_size = 77

#For 216M OpenWebText Pre-trained model
TOKEN_MODE = "pile"
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
vocab_size = 50277

MODEL_NAME = 'sst2_model_ten'
n_layer = 18
n_embd = 768
ctx_len = 1024

args.MODEL_NAME = MODEL_NAME
args.n_layer = n_layer
args.n_embd = n_embd
args.ctx_len = ctx_len
args.vocab_size = vocab_size
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

B = 4
T = 5


# Generate random input indices (tokens) in the vocabulary range
#idx = torch.randint(0, vocab_size, (B, T), dtype=torch.long)

# Optionally, generate random target indices for loss calculation
#targets = torch.randint(0, vocab_size, (B, T), dtype=torch.long)

#print(idx)
#print(targets)



#pbar = gen_pbar()


from src.model_run import RWKV_RNN
from src.class_model import GPT, GPTConfig

model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, num_classes=2, model_type='RWKV', n_layer=18, n_embd=768))
quant_model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, num_classes=2, model_type='RWKV', n_layer=18, n_embd=768))
m2 = torch.load(MODEL_NAME + '.pth', map_location=torch.device('cpu'))


model.load_state_dict(m2)
model = model.cuda()
model.eval()

print("param name\t param shape\t param type\t num of elements\t param size (bytes)")
for key, param in m2.items():
    print(f"{key}\t{param.size()}\t{param.dtype}\t{param.nelement()}\t{param.nelement() * param.element_size()}")
    #param.cpu()
exit()
