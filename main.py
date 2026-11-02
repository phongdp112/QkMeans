from dataset import Dataset
from QKmeans import QKmeans
from sklearn.cluster import KMeans, kmeans_plusplus

if __name__ == "__main__":
    ds = Dataset('iris', 'ISP')
    conf = {
        'quantization': 1,      # 1 = q1:1, 2 = q1:k, 3 = qM:k
        'shots': 1024,
        'sc_tresh': 1e-3,       # 0 thì không bao giờ dừng sớm
        'K': 3,                 # đối chiếu tên khóa trong __init__ của QKmeans.py
        'random_init_centroids': False
    }
    seed = 123
    if conf['random_init_centroids']:
            initial_centroids = ds.original_df.sample(n=conf['K'], random_state=seed).values
    else:
        initial_centroids, indices = kmeans_plusplus(ds.original_df.values, n_clusters=conf['K'], random_state=seed)
    
    qk = QKmeans(ds, conf)
    qk.run(initial_centroids)
