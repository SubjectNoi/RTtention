import torch
import sys
from torch.profiler import profile, record_function, ProfilerActivity
# Baseline: 1048576 128 = 298us
SEQ_LEN, HEAD_DIM = int(sys.argv[1]), int(sys.argv[2])
dev = "cuda:0"
IN = torch.randn((1, HEAD_DIM)).type(torch.float16).to(dev)
WEIGHT = torch.randn((HEAD_DIM, SEQ_LEN)).type(torch.float16).to(dev)
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
	with record_function("torch_gemv"):
		OUT = torch.matmul(IN, WEIGHT)

print(prof.key_averages().table())
