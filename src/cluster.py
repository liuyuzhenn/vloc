from loc.kmeans import kmeans, kmeans
import argparse
from recon.database import DatabaseOperator
import time
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str, required=True,
                    help='Path to database file')
parser.add_argument('--output', type=str, required=True,
                    help='Number of descriptors per image')
parser.add_argument('--num_clusters', type=int, required=True,
                    help='Number of clusters')
parser.add_argument('--method', type=str, default='kmeans',
                    help='Method for clustering')
parser.add_argument('--num_descs', type=int, default=2000,
                    help='Number of descriptors per image')
parser.add_argument('--iterations', type=int, default=1000,
                    help='Maximum iterations of kmeans')
args = parser.parse_args()
db = args.database
out = args.output
method = args.method
k = args.num_clusters
num_descs = args.num_descs
iters = args.iterations
eps = 1e-3

# example
# db = "../data/words_alike.db"
# out = "../data/words_alike.npy"
# method = 'kmeans'
# k = 2000
# iters = 1000
# num_descs = 2000
# eps = 0.001

database = DatabaseOperator(db)
descriptors = database.fetch_all_descriptors()
descriptors = np.concatenate([d[1][np.random.choice(len(d[1]), num_descs, replace=False)] for d in descriptors]).astype(np.float32)
t1 = time.time()
if method=='kmeans':
    centers = kmeans(descriptors, k=k, iterations=iters, eps=eps)
else:
    raise NotImplementedError
t2 = time.time()
print('K-means time cost is {:.1f} s'.format(t2-t1))
np.save(out, centers)
