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



#########################################################################
#Quantize Weights Only (Linear)
#########################################################################


def basic_quantize(weights, bits=8):
    with torch.no_grad():
        scale = (torch.max(weights).item()-torch.min(weights).item())/((2**(bits-1)-1)-(-2**(bits-1)-1))
        print(scale)
        x = weights / scale
        y = torch.round(x)
        y = y.int()
        return y

def get_scale(weights, bits=8):
    scale = (torch.max(weights).item()-torch.min(weights).item())/((2**(bits-1)-1)-(-2**(bits-1)-1))
    #print(scale)
    return scale


epoch_length_fixed = 100
datafile_test = 'wikitext-103.test_text_document'

test_dataset = Dataset(MMapIndexedDataset(datafile_test), 100, epoch_length_fixed)
print(test_dataset)
loader = DataLoader(test_dataset, shuffle=False, batch_size=4)





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

MODEL_NAME = 'SpikeGPT-216M'
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

from src.model_run import RWKV_RNN
from src.model import GPT, GPTConfig

model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, model_type='RWKV', n_layer=18, n_embd=768))
quant_model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, model_type='RWKV', n_layer=18, n_embd=768))
m2 = torch.load(MODEL_NAME + '.pth', map_location=torch.device('cpu'))

#pbar = gen_pbar()
param_list = ['receptance.weight', 'key.weight', 'value.weight']




model.load_state_dict(m2)
model = model.cuda()
model.eval()


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
bypass_selection = False

print("Non-Quantized Perplexity, Quantized Perplexity")
for i in range(1):
    for item in float_dict:
        if any(x in item for x in param_list) or bypass_selection:
            #print(f"block.{i}")
            with torch.no_grad():
                #print(item)
                weights = float_dict[item]
                q_weights = quantize_weights(weights)
                float_dict[item] = q_weights  # Now contains quantization metadata
                print(float_dict[item].dtype)
                print(float_dict[item])

    model.load_state_dict(m2)
    model = model.cuda()
    model.eval()
    torch.save(float_dict, "test_quant.pth")

    quant_dict = torch.load("test_quant.pth")
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

    for name, param in quant_model.named_parameters():
        if any(x in name for x in param_list) or bypass_selection:
            print(param)
            print(param.dtype)
        break


    torch.save(quant_model.state_dict(), "test.pth")

    quant_model.cuda()


    original_tens = []
    quant_tens = []



    for (x,y) in loader:
        out = model(x, y)
        out = torch.square(out)
        functional.reset_net(model)
        original_tens.append(out.item())
        #print("original: ",out)
        out = quant_model(x, y)
        out = torch.square(out)
        functional.reset_net(quant_model)
        quant_tens.append(out.item())
        #print("quantized",out)
        #exit()
    print(f"{statistics.mean(original_tens)}, {statistics.mean(quant_tens)}")
    #print('Quantized perplexity: ', )


