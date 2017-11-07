from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

# @build:deps [
# @/deeplearning/projects/faiss:pyfaiss
# ]
# also in fbsource/fbcode/fblearner/flow/projects/rl/TARGETS
# but no need in fbsource/fbcode/fblearner/flow/facebook/canary/TARGETS
# if need to include faiss_gpu:
# # @/deeplearning/projects/faiss:pyfaiss_gpu

import faiss


# Note: this is a simple replacement for np.unique(a, axis=0) since
#  axis is supported after numpy 1.3.7 where ours is 1.3.1
def unique_rows(data):
    sorted_data = data[np.lexsort(data.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_data, axis=0), 1))
    return sorted_data[row_mask]


# https://stackoverflow.com/questions/14766194/
def contain_row(data, row):
    return any(np.equal(data, row).all(1))


def lookup_row(data, row):
    return np.where((data == row).all(axis=1))[0].tolist()


def fetch_new_rows_toappend(data, newrows):
    def not_contained_in_data(row):
        return not contain_row(data, row)

    row_mask = np.apply_along_axis(not_contained_in_data, 1, newrows)
    return newrows[row_mask]


def act_hash(act_vector):
    # temporarily using the array itself
    return str(act_vector)


class KnnEngine(object):
    def __init__(self, data_dim, data_size_limit, k=1, index_type='IP'):
        self._knn_indextype = index_type
        # 'L2' or 'IP' # for inner product
        self._data_dim = data_dim
        self._data_size_limit = data_size_limit
        self._knn_search_index = None
        self._knn_search_index_quatizer = None
        self._knn_search_k = k
        self._knn_default_n_centroid = 100
        self._knn_datasize_threshold_faster = 500000
        self._knn_subsampling_training = 0.05  # subsampling only if positive

    # benchmark:
    # full: https://github.com/facebookresearch/faiss/blob/master/benchs/README.md
    # 1m https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors
    # 1mgpu https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py

    def build_index_knn(self, all_data):
        all_data = all_data.astype(np.float32)

        if all_data.shape[0] <= self._knn_datasize_threshold_faster:
            self._knn_search_index = faiss.IndexFlatIP(self._data_dim) \
                if self._knn_indextype == 'IP' \
                else faiss.IndexFlatL2(self._data_dim)
        # check if data count worth training of cluster
        if all_data.shape[0] > self._knn_datasize_threshold_faster:
            # try:
            # index_type = faiss.METRIC_IP if self._knn_indextype == 'IP' else \
            #     faiss.METRIC_L2
            target_dim = self._data_dim
            self._knn_search_index_quatizer = faiss.IndexFlatIP(target_dim) \
                if self._knn_indextype == 'IP' \
                else faiss.IndexFlatL2(target_dim)
            self._knn_search_index = faiss.IndexIVFPQ(
                self._knn_search_index_quatizer, target_dim, 16, 8, 8
            )
            # faiss.IndexIVFFlat(coarseQuantizer, self._data_dim, nlist, index_type)
            self._knn_search_index.do_polysemous_training = True
            self._knn_search_index.verbose = True
            training_dataset = None
            if self._knn_subsampling_training > 0:
                trained_w10per = np.random.choice(
                    len(all_data),
                    int(round(len(all_data) * self._knn_subsampling_training))
                )
                training_dataset = all_data[trained_w10per, :]
            else:
                training_dataset = all_data
            print("training vectors to index")
            self._knn_search_index.train(training_dataset)
            # finally:
            #     print(
            #         "Warning: training tree failed, but still with regular knn tree"
            #     )

        self._knn_search_index.add(all_data)
        return True

    def append_index_knn(self, append_more_data):
        assert self._knn_search_index.is_trained
        self._knn_search_index.add(append_more_data.astype(np.float32))

    def find_knn_dist_ind(self, data_query):
        assert self._knn_search_index.is_trained
        dist, indx = self._knn_search_index.search(
            data_query.astype(np.float32), k=self._knn_search_k
        )
        # return k for each data in query, if q special case only need 1
        return dist, indx

    def find_knn_best_dist_ind(self, data_query):
        dist, indx = self.find_knn_dist_ind(data_query)
        if self._knn_search_k == 1:
            return dist.flatten(), indx.flatten()
        dist_slicebyk = dist.reshape((self._knn_search_k, -1))
        indx_slicebyk = indx.reshape((self._knn_search_k, -1))
        best_slice = np.argmax(dist_slicebyk, axis=0)
        best_dist = dist_slicebyk[best_slice, range(best_slice.shape[0])]
        best_indx = indx_slicebyk[best_slice, range(best_slice.shape[0])]
        return best_dist, best_indx

    def check_engine_build(self):
        return self._knn_search_index is not None
