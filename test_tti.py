import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import faiss
from faiss.contrib.evaluation import knn_intersection_measure

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.uint32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.uint32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

d = 200
n = 1000000
q = 1000

xb = read_fbin ("/home/wennitao/workspace/tti1M/base.1M.fbin")
xq = read_fbin ("/home/wennitao/workspace/tti1M/query.public.100K.fbin")
# gt = read_ibin ("/home/wennitao/workspace/tti1M/groundtruth.public.100K.ibin")

print (xb.shape, xq.shape)

res = faiss.StandardGpuResources()
# index = faiss.index_factory(d, "IVF4096,Flat", faiss.METRIC_INNER_PRODUCT)
index = faiss.index_factory(d, "IVF4096,PQ40x12", faiss.METRIC_INNER_PRODUCT)
co = faiss.GpuClonerOptions()
co.useFloat16 = True

# index = faiss.index_cpu_to_gpu(res, 0, index, co)
# index.train (xb)
# index.add (xb)

# index = faiss.index_gpu_to_cpu (index)
# faiss.write_index (index, "/home/wennitao/workspace/tti1M/index.IVF4096.PQ40x12.1M.index")

index = faiss.read_index ("/home/wennitao/workspace/tti1M/index.IVF4096.PQ40x12.1M.index")
# index = faiss.read_index ("/home/wennitao/workspace/tti1M/index.1M.index")
# index = faiss.index_cpu_to_gpu(res, 0, index, co)

gt_index = faiss.IndexFlatIP(d)
gt_index = faiss.index_cpu_to_gpu(res, 0, gt_index, co)
gt_index.add(xb)

index.nprobe = 256
D, I = index.search(xq[:q], 100)
gt_D, gt_I = gt_index.search(xq[:q], 100)

# Recall 1@100
recall = 0
for i in range (q):
    recall += gt_I[i][0] in I[i]
print (recall / 1000)