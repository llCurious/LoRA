import time
import math
import torch

B = 8
SEQ_LEN = 512
r = 4

# [base, medium, large]
MODEL = "medium"

if MODEL == "base":
    n_embd = 768
    n_head = 12
elif MODEL == "medium":
    n_embd = 1024
    n_head = 16
elif MODEL == "large":
    n_embd = 1280
    n_head = 20
else:
    raise NotImplementedError(f"Model {MODEL} not supported.")

# GPU linear time: 0.19550323486328125 ms
# CPU linear time: 9.151220321655273 ms
# CPU lora time: 1.3489723205566406 ms
def attention_qkv():
    x = torch.rand(size=(B, SEQ_LEN, n_embd))
    
    # GPU computations
    start = time.time()
    x = x.cuda()
    print(f"GPU qkv io time: {(time.time()-start)*1000} ms")
    gpu_linear = torch.nn.Linear(n_embd, n_embd).cuda()

    start = time.time()
    gpu_res = gpu_linear(x)
    end = time.time()
    print(f"GPU linear time: {(end-start)*1000} ms")

    x = x.cpu()
    cpu_linear = gpu_linear.cpu()
    start = time.time()
    cpu_res = cpu_linear(x)
    end = time.time()
    print(f"CPU linear time: {(end-start)*1000} ms")

    # cpu lora computations
    x = x.cpu()
    lora_A = torch.nn.Parameter(torch.rand((r, n_embd)))
    lora_B = torch.nn.Parameter(torch.rand((n_embd, r)))
    start = time.time()
    cpu_lora = x @ lora_A.transpose(0, 1) @ lora_B.transpose(0, 1)
    end = time.time()
    print(f"CPU lora time: {(end-start)*1000} ms")

# Same as qkv
def attention_o():
    x = torch.rand(size=(B, SEQ_LEN, n_embd))
    
    # GPU computations
    x = x.cuda()
    gpu_linear = torch.nn.Linear(n_embd, n_embd).cuda()
    start = time.time()
    gpu_res = gpu_linear(x)
    end = time.time()

    x = x.cpu()
    cpu_linear = gpu_linear.cpu()
    start = time.time()
    cpu_res = cpu_linear(x)
    end = time.time()
    print(f"CPU linear time: {(end-start)*1000} ms")

    # cpu lora computations
    x = x.cpu()
    lora_A = torch.nn.Parameter(torch.rand((r, n_embd)))
    lora_B = torch.nn.Parameter(torch.rand((n_embd, r)))
    start = time.time()
    cpu_lora = x @ lora_A.transpose(0, 1) @ lora_B.transpose(0, 1)
    end = time.time()
    print(f"CPU lora time: {(end-start)*1000} ms")

def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )

def attention_mlp():
    x = torch.rand(size=(B, SEQ_LEN, n_embd))
    
    # GPU computations
    start = time.time()
    x = x.cuda()
    print(f"GPU mlp io time: {(time.time()-start)*1000} ms")

    gpu_linear1 = torch.nn.Linear(n_embd, n_embd*4).cuda()
    gpu_linear2 = torch.nn.Linear(n_embd*4, n_embd).cuda()

    act = gelu_impl

    start = time.time()
    h1 = gpu_linear1(x)
    end = time.time()
    print(f"fc1 GPU time {(end - start) * 1000} ms")

    start = time.time()
    h1 = act(h1)
    print(h1.shape)
    end = time.time()
    print(f"act GPU time {(end - start) * 1000} ms")


    start = time.time()
    # topk, indices = torch.topk(h1, k = 128, dim=-1)
    # h1 = torch.zeros_like(h1).scatter(-1, indices, topk)
    # h1 = h1.to_sparse_csc()
    # h1 = h1.cpu().to_dense()
    h1 = h1.cpu()
    end = time.time()
    print(f"GPU -> CPU IO time {(end - start) * 1000} ms")


    start = time.time()
    h1 = act(h1)
    end = time.time()
    print(f"act cpu time {(end - start) * 1000} ms")

    
    start = time.time()
    h1 = h1.cuda()
    end = time.time()
    print(f"CPU -> GPU IO time {(end - start) * 1000} ms")

    start = time.time()
    gpu_res = gpu_linear2(h1)
    end = time.time()
    print(f"fc2 GPU time: {(end-start)*1000} ms")

    x = x.cpu()
    cpu_linear1 = gpu_linear1.cpu()
    cpu_linear2 = gpu_linear2.cpu()
    start = time.time()
    cpu_res = cpu_linear2(cpu_linear1(x))
    end = time.time()
    print(f"CPU linear time: {(end-start)*1000} ms")

    # cpu lora computations
    x = x.cpu()
    lora_A = torch.nn.Parameter(torch.rand((r, n_embd)))
    lora_B = torch.nn.Parameter(torch.rand((n_embd*4, r)))

    lora_A2 = torch.nn.Parameter(torch.rand((r, n_embd*4)))
    lora_B2 = torch.nn.Parameter(torch.rand((n_embd, r)))
    start = time.time()
    cpu_lora = x @ lora_A.transpose(0, 1) @ lora_B.transpose(0, 1)
    cpu_lora_res = cpu_lora @ lora_A2.transpose(0, 1) @ lora_B2.transpose(0, 1)
    end = time.time()
    print(f"CPU lora time: {(end-start)*1000} ms")

print(f"qkv statistics")
attention_qkv()

print(f"mlp statistics")
attention_mlp()
