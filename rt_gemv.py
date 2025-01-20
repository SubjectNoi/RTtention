import faiss
import torch
from RTtention import rt_gemv
from faiss.contrib.inspect_tools import get_pq_centroids
# (1, 128).dot((1048576, 128))

# first train a codebook on (:, 128)
dev = "cuda:0"

# If three .pt file are not exist
# weight = torch.randn((1048576, 128)).type(torch.float16).to(dev)
# index = faiss.index_factory(128, "PQ64x8np")
# index.train(weight.to("cpu").numpy())
# codebook = torch.from_numpy(get_pq_centroids(index.pq)).type(torch.float16).to(dev)
# weight_quantized = torch.from_numpy(index.pq.compute_codes(weight.to("cpu").numpy())).to(dev)
# weight_dequantized = torch.from_numpy(index.pq.decode(weight_quantized.to("cpu").numpy())).to(dev)
# torch.save(weight, "/home/zhliu/workspace/RTtention/weight.pt")
# torch.save(weight_quantized, "/home/zhliu/workspace/RTtention/weight_quantized.pt")
# torch.save(codebook, "/home/zhliu/workspace/RTtention/codebook.pt")
# torch.save(weight_dequantized, "/home/zhliu/workspace/RTtention/weight_dequantized.pt")

weight = torch.load("/home/zhliu/workspace/RTtention/weight.pt").to(dev)
weight_quantized = torch.load("/home/zhliu/workspace/RTtention/weight_quantized.pt").to(dev)
codebook = torch.load("/home/zhliu/workspace/RTtention/codebook.pt").to(dev)
weight_dequantized = torch.load("/home/zhliu/workspace/RTtention/weight_dequantized.pt").type(torch.float16).to(dev)
IN = torch.randn((1, 128)).type(torch.float16).to(dev)

# print(codebook.shape)

# # Reference, pick top-65536 (6.25%)
# OUT_REF = torch.matmul(IN, weight_dequantized.transpose(0, 1))
OUT = rt_gemv(IN, weight_quantized, codebook)


