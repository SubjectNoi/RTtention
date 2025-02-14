import faiss
import torch
from RTtention import rt_gemv
from faiss.contrib.inspect_tools import get_pq_centroids
# (1, 128).dot((1048576, 128))

# first train a codebook on (:, 128)
dev = "cuda:0"

seq_len = 65536
head = 32
dim = 128

# If three .pt file are not exist
weight = torch.randn((seq_len, dim)).type(torch.float16)
# index = faiss.index_factory(dim, "PQ64x8np")
# index.train(weight.to("cpu").numpy())
# codebook = torch.from_numpy(get_pq_centroids(index.pq)).type(torch.float16)
# weight_quantized = torch.from_numpy(index.pq.compute_codes(weight.to("cpu").numpy()))
# weight_dequantized = torch.from_numpy(index.pq.decode(weight_quantized.to("cpu").numpy()))
# torch.save(weight, "/home/wennitao/workspace/RTtention/weight.pt")
# torch.save(weight_quantized, "/home/wennitao/workspace/RTtention/weight_quantized.pt")
# torch.save(codebook, "/home/wennitao/workspace/RTtention/codebook.pt")
# torch.save(weight_dequantized, "/home/wennitao/workspace/RTtention/weight_dequantized.pt")

# weight = torch.load("/home/wennitao/workspace/RTtention/weight.pt")
# weight_quantized = torch.load("/home/wennitao/workspace/RTtention/weight_quantized.pt")
# codebook = torch.load("/home/wennitao/workspace/RTtention/codebook.pt")
# weight_dequantized = torch.load("/home/wennitao/workspace/RTtention/weight_dequantized.pt").type(torch.float16)

q = torch.randn ((head, dim)).type (torch.float16)
k_cache = torch.randn ((seq_len, head, dim))

idx = torch.arange(seq_len * head * dim // 2, dtype=torch.int32)
levels = idx // (dim // 2)
entries = idx % (dim // 2)

centers = torch.zeros ((seq_len * head * dim // 2, 3), dtype=torch.float32).to(dev)
centers[idx, 0] = k_cache[levels // head, levels % head, entries * 2].to (dev)
centers[idx, 1] = k_cache[levels // head, levels % head, entries * 2 + 1].to (dev)
centers[idx, 2] = (levels * 2 + 1).float().to(dev)

RADIUS = 0.2
# radius = torch.sqrt(RADIUS ** 2 + centers[idx, 0] ** 2 + centers[idx, 1] ** 2).type (torch.float32).to(dev)
radius = torch.zeros ((seq_len * head * dim // 2), dtype=torch.float32).to(dev)
radius = torch.fill (radius, RADIUS).to(dev)

idx = torch.arange(head * dim // 2, dtype=torch.int32)
origins = torch.zeros ((head * dim // 2, 3), dtype=torch.float32).to(dev)
origins[idx, 0] = q[idx // (dim // 2), idx % (dim // 2) * 2].float().to (dev)
origins[idx, 1] = q[idx // (dim // 2), idx % (dim // 2) * 2 + 1].float().to (dev)
origins[idx, 2] = (idx * 2).float().to(dev)

print (centers.shape, radius.shape, origins.shape)

OUT = rt_gemv(q, weight, weight, centers, radius, origins)

# IN = torch.randn((1, 128)).type(torch.float16).to(dev)

# SPACE = 64
# ENTRY = 256
# # centers = torch.zeros ((64 * 256 * 3), dtype=torch.float32).to(dev)
# # Compute idx tensor (flattened indices)
# idx = torch.arange(SPACE * ENTRY, dtype=torch.int32)

# # Compute level and entry indices
# levels = idx // ENTRY
# entries = idx % ENTRY

# print (codebook.shape)

# # Compute x, y, z
# x = codebook[levels, entries, 0]
# y = codebook[levels, entries, 1]
# z = levels * 2 + 1  # Vectorized calculation for z
# z = z.to(dev)

# # Compute centers
# centers = torch.zeros((SPACE * ENTRY, 3), dtype=torch.float32).to(dev)
# centers[idx, 0] = x.float()
# centers[idx, 1] = y.float()
# centers[idx, 2] = z.float()

# RADIUS = 0.5

# # Compute radius
# radius = torch.sqrt(RADIUS ** 2 + x ** 2 + y ** 2).type (torch.float32).to(dev)

# idx = torch.arange(SPACE, dtype=torch.int32)
# origins = torch.zeros((SPACE, 3), dtype=torch.float32).to(dev)
# origins[idx, 0] = IN[:, idx * 2].type (torch.float32)
# origins[idx, 1] = IN[:, idx * 2 + 1].type (torch.float32)
# origins[idx, 2] = (idx * 2).type (torch.float32).to (dev)

# # # Reference, pick top-65536 (6.25%)
# # OUT_REF = torch.matmul(IN, weight_dequantized.transpose(0, 1))
# OUT = rt_gemv(IN, weight_quantized, codebook, centers, radius, origins)


