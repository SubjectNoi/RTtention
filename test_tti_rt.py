import numpy as np
import sys
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from cuml import KMeans
import matplotlib.pyplot as plt
import faiss
import os
import math
import time

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
q = 100

xb = read_fbin ("/home/wennitao/workspace/tti1M/base.1M.fbin")
xq = read_fbin ("/home/wennitao/workspace/tti1M/query.public.100K.fbin")
# gt = read_ibin ("/home/wennitao/workspace/tti1M/groundtruth.public.100K.ibin")

print (xb.shape, xq.shape)

stat = []
for i in range(d):
    stat.append([i, np.min(xb[:, i]), np.max(xb[:, i]), np.mean(xb[:, i]), np.std(xb[:, i])])

nlists = 100
nprobe = 32

# kmeans = KMeans(n_clusters=nlists, init='scalable-k-means++', n_init=32, max_iter=600).fit(xb)
# cluster_centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

# cluster_centroids = np.array (cluster_centroids, dtype=np.float32)
# labels = np.array (labels, dtype=np.int32)

# np.save ("/home/wennitao/workspace/tti1M/100_cluster_centroids.npy", cluster_centroids)
# np.save ("/home/wennitao/workspace/tti1M/100_labels.npy", labels)

cluster_centroids = np.load ("/home/wennitao/workspace/tti1M/100_cluster_centroids.npy")
labels = np.load ("/home/wennitao/workspace/tti1M/100_labels.npy")

print ("1st cluster finished. ")

cluster_points = {}
for i in range (nlists):
    cluster_points[i] = xb[labels == i]

# for i in range (n):
#     if labels[i] not in cluster_points:
#         cluster_points[labels[i]] = []
#     cluster_points[labels[i]].append (xb[i])

# for cluster_id in range (nlists):
#     cluster_points[cluster_id] = np.array (cluster_points[cluster_id])
    # print (cluster_id, cluster_points[cluster_id].shape)

m = 100
nbits = 8
pq_d = int (d // m)

max_dist = []
for i in range(m):
    res = 0.0
    for j in range(pq_d):
        res += (stat[i * pq_d + j][2] ** 2)
    max_dist.append(res ** 0.5)

codebook = {}

# for cluster_id in range (nlists):
#     for d in range (m):
#         cur_cluster_points = cluster_points[cluster_id]
#         subdim_cluster_points = cur_cluster_points[:, d*pq_d:(d+1)*pq_d]
#         kmeans = KMeans(n_clusters=min (2 ** nbits, subdim_cluster_points.shape[0]), init='scalable-k-means++', n_init=32, max_iter=600).fit(subdim_cluster_points)
#         sub_cluster_centroids = kmeans.cluster_centers_
#         sub_labels = kmeans.labels_
#         sub_cluster_centroids = np.array (sub_cluster_centroids, dtype=np.float32)
#         sub_labels = np.array (sub_labels, dtype=np.int32)
#         if os.path.exists ("/home/wennitao/workspace/tti1M/cluster_%d" % cluster_id) == False:
#             os.makedirs ("/home/wennitao/workspace/tti1M/cluster_%d" % cluster_id)
#         np.save ("/home/wennitao/workspace/tti1M/cluster_%d/sub_%d_centroids.npy" % (cluster_id, d), sub_cluster_centroids)
#         np.save ("/home/wennitao/workspace/tti1M/cluster_%d/sub_%d_labels.npy" % (cluster_id, d), sub_labels)

# from multiprocessing import Pool
# with Pool(32) as p:
#     p.map(buildPQ, range (nlists * m))

# PQ on all residual
# residual = np.zeros ((n, d), dtype=np.float32)
# for cluster_id in range (nlists):
#     residual[labels == cluster_id] = xb[labels == cluster_id] - cluster_centroids[cluster_id]

# for d in range (m):
#     subdim_residual = residual[:, d*pq_d:(d+1)*pq_d]
#     kmeans = KMeans(n_clusters=2 ** nbits, init='k-means++', n_init=32, max_iter=600).fit(subdim_residual)
#     sub_cluster_centroids = kmeans.cluster_centers_
#     sub_labels = kmeans.labels_
#     sub_cluster_centroids = np.array (sub_cluster_centroids, dtype=np.float32)
#     sub_labels = np.array (sub_labels, dtype=np.int32)
#     np.save ("/home/wennitao/workspace/tti1M/PQ100x8/sub_%d_centroids.npy" % d, sub_cluster_centroids)
#     np.save ("/home/wennitao/workspace/tti1M/PQ100x8/sub_%d_labels.npy" % d, sub_labels)

quantized = np.zeros ((n, m), dtype=np.int32)

for cluster_id in range (nlists):
    sub_centroids = []
    sub_labels = []
    for dim in range (m):
        sub_centroids.append (np.load ("/home/wennitao/workspace/tti1M/cluster_%d/sub_%d_centroids.npy" % (cluster_id, dim)))
        sub_labels.append (np.load ("/home/wennitao/workspace/tti1M/cluster_%d/sub_%d_labels.npy" % (cluster_id, dim)))
    codebook[cluster_id] = np.array (sub_centroids)
    quantized[labels == cluster_id] = np.array ([sub_labels[dim] for dim in range (m)]).T

print ("2nd cluster finished. ")

# for dim in range (m):
#     sub_centroids = np.load ("/home/wennitao/workspace/tti1M/PQ100x8/sub_%d_centroids.npy" % dim)
#     sub_labels = np.load ("/home/wennitao/workspace/tti1M/PQ100x8/sub_%d_labels.npy" % dim)
#     codebook[dim] = sub_centroids
#     quantized[:, dim] = sub_labels

recall = 0

# ground truth
res = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()
co.useFloat16 = True
gt_index = faiss.IndexFlatIP(d)
gt_index = faiss.index_cpu_to_gpu(res, 0, gt_index, co)
gt_index.add(xb)
gt_D, gt_I = gt_index.search(xq[:q], 100)

# def checkIntersect (x, q, r):
#     r = math.sqrt (r ** 2 + x[0] ** 2 + x[1] ** 2)
#     if (q[0] - x[0]) ** 2 + (q[1] - x[1]) ** 2 <= r ** 2:
#         return np.dot (x, q)
#     else:
#         return -1e10

def checkIntersect (query, sub_centroids, radius, max_dist):
    # query: (pq_d, )
    # sub_centroids: (2 ** nbits, pq_d)
    # radius: float
    # return: (2 ** nbits, )
    hit_count = 0
    dis = np.zeros ((sub_centroids.shape[0], ), dtype=np.float32)
    for i in range (sub_centroids.shape[0]):
        if (query[0] - sub_centroids[i, 0]) ** 2 + (query[1] - sub_centroids[i, 1]) ** 2 <= ((radius * max_dist) ** 2 + sub_centroids[i, 0] ** 2 + sub_centroids[i, 1] ** 2):
            hit_count += 1
            dis[i] = np.dot (query, sub_centroids[i])
        else:
            dis[i] = -max_dist
    # dis = np.dot (query, sub_centroids.T)
    # dis[np.sqrt ((query[0] - sub_centroids[:, 0]) ** 2 + (query[1] - sub_centroids[:, 1]) ** 2) > (radius * max_dist) ** 2] = -max_dist
    # print ("hit_count: ", hit_count)
    # print ("hit_rate: ", hit_count / sub_centroids.shape[0])
    return dis, hit_count

radius = 0.7

def IVFPQ(query_id):
    # IVFPQ
    # for query_id in range (q):
    hit_count = 0
    query = xq[query_id]
    final_dis = np.full ((n, ), -1e10, dtype=np.float32)
    chosen_clusters = np.argsort (np.dot (cluster_centroids, query))[-nprobe:]

    for cluster_id in chosen_clusters:
        dis = np.zeros ((cluster_points[cluster_id].shape[0], ), dtype=np.float32)
        sub_labels = quantized[labels == cluster_id]
        # cur_query = query - cluster_centroids[cluster_id]
        cur_query = query
        for dim in range (m):
            sub_query = cur_query[dim*pq_d:(dim+1)*pq_d]
            sub_centroids = codebook[cluster_id][dim] # 2 ** nbits

            # intersect query with sub_centroids
            sub_centroids_dis, cur_hit_count = checkIntersect (sub_query, sub_centroids, radius, max_dist[dim])
            dis += sub_centroids_dis[sub_labels[:, dim]]
            hit_count += cur_hit_count

            # direct compute
            # dis += np.dot (sub_query, sub_centroids[sub_labels[:, dim]].T)
        final_dis[labels == cluster_id] = np.maximum (final_dis[labels == cluster_id], dis)

    print ("hit count rate: ", hit_count / (nprobe * m * 2 ** nbits))

    final_I = np.argsort (-final_dis)
    final_I = final_I[:100]
    return gt_I[query_id][0] in final_I

from multiprocessing import Pool
with Pool(32) as p:
    res = p.map(IVFPQ, range(q))

# IVFPQ (0)
recall = sum (res)
print (recall / q)