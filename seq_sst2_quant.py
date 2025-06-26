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
args.RUN_DEVICE = "cpu" # 'cuda' // 'cpu' (already fast)
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

model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, model_type='RWKV', n_layer=18, n_embd=768))
quant_model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, model_type='RWKV', n_layer=18, n_embd=768))
m2 = torch.load(MODEL_NAME + '.pth', map_location=torch.device('cpu'))


model.load_state_dict(m2)
model = model.cuda()
model.eval()

optimizer = AdamW(model.parameters(), lr=3e-6)
loss_fn = nn.CrossEntropyLoss()

#########################################################################
#Quantize Weights Only
#########################################################################


# Modified quantization function
def quantize_weights(weights, bits=8):
    q_min = -2**(bits-1)
    q_max = 2**(bits-1)-1
    # Calculate scale and zero point
    scale = (weights.max() - weights.min()) / (q_max - q_min)
    zero_point = torch.round((q_min - weights.min()/scale)).clamp(q_min, q_max)
    zero_point = zero_point.int()

    #assigns dtype based on required bandwith
    q_type = torch.qint8 if bits<=8 else torch.qint32

    # Quantize using PyTorch's native functions
    q = torch.quantize_per_tensor(
        weights,
        scale=scale.item(),
        zero_point=zero_point.item(),
        dtype=q_type
    )

    return q



float_dict = model.state_dict()
model.load_state_dict(m2)
model = model.cuda()
model.eval()


#Amend parameter list to select which type to be quantized (or carry out full quantization by changing bypass_selection)
param_list = ['receptance.weight', 'key.weight', 'value.weight', 'ln1', 'ln2']
bits = [4, 8, 12, 16, 32]
bypass_selection = True

#print("Non-Quantized Perplexity, Quantized Perplexity, Deviation (%), More or Less Perplexity")
#Open file for recording
rec_file = open("quanti_sst2_all.txt", 'a')
loss_file = open("orig_sst2_all.txt", 'a')
rec_file.write("Epoch, 4-bit, 8-bit, 12-bit, 16-bit, 32-bit\n")
for i in range(25):
    rec_file.write(f"{i},")
    loss_file.write(f"{i},")
    for b in bits:
        for item in float_dict:
            if any(x in item for x in param_list) or bypass_selection:
                #print(f"block.{i}")
                with torch.no_grad():
                    #print(item)
                    weights = float_dict[item]
                    q_weights = quantize_weights(weights, bits=b)
                    float_dict[item] = q_weights  # Now contains quantization metadata
                    #print(float_dict[item].dtype)
                    #print(float_dict[item])



        torch.save(float_dict, "test_quant_sst.pth")

        #quant_dict = torch.load("test_quant.pth")
        quant_dict = float_dict
        with torch.no_grad():
            for item in quant_dict:
                #print(float_dict[item])
                quant_dict[item] = torch.dequantize(quant_dict[item])


        quant_model.load_state_dict(quant_dict)
        float_dict = model.state_dict()

        #for param in model.parameters():
        #    with torch.no_grad():
        #        param.data = param.data.to(torch.half)
            #print(param)
            #print(param.dtype)

        #for name, param in quant_model.named_parameters():
        #    if any(x in name for x in param_list) or bypass_selection:
        #        print(param)
        #        print(param.dtype)
        #    break


        #torch.save(quant_model.state_dict(), "test.pth")

        quant_model.cuda()


        original_tens = []
        quant_tens = []

        total_loss = 0
        total_correct = 0
        total_samples = 0
        q_total_loss = 0
        q_total_correct = 0
        q_total_samples = 0

        for batch in tqdm(test_loader):
            # Get the inputs and labels
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass
            with torch.no_grad():
                outputs, loss = model(inputs)
                #print(outputs.size())
                pred = outputs.argmax(dim=1)
                loss = loss_fn(outputs, labels)
                q_out, q_loss = quant_model(inputs)
                q_pred = q_out.argmax(dim=1)
                q_loss = loss_fn(q_out, labels)


            # Update the metrics
            total_loss += loss.item() * inputs.size(0)
            total_correct += (pred == labels).sum().item()
            total_samples += inputs.size(0)
            q_total_loss += q_loss.item() * inputs.size(0)
            q_total_correct += (q_pred == labels).sum().item()
            q_total_samples += inputs.size(0)
            functional.reset_net(model)
            functional.reset_net(quant_model)
            #break

        # Compute the metrics
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        q_avg_loss = q_total_loss / q_total_samples
        q_avg_acc = q_total_correct / q_total_samples


        # Print the metrics
        print(f"Original: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")
        print(f"Quantized {b}-bits: Loss = {q_avg_loss:.4f}, Accuracy = {q_avg_acc:.4f}\n")
        rec_file.write(f"{q_avg_acc:.4f},")
        loss_file.write(f"{avg_acc:.4f},")
    rec_file.write("\n")
    loss_file.write("\n")
    torch.save(model.state_dict(), "sst2_model_ten.pth")
rec_file.close()
loss_file.close()

