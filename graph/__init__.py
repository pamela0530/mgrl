import pickle
import scipy.sparse as sp
import numpy as np
import torch

def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(
            (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class BasicGraph:

    def __init__(self, objects_dic =None):
        self.word_vectors = None
        self.edges_weight= {}
        self.objects = objects_dic

    def read_edge_from_json(self,file_name):
        f = open(file_name, "rb")
        self.edges_weight = pickle.load(f)
        f.close()

    def read_node_from_json(self,file_name):
        f = open(file_name, "rb")
        self.word_vectors = pickle.load(f)
        f.close()

    def save_graph(self,file_path):
        edges = pickle.dumps(self.edges_weight)
        with open(file_path, "wb") as f:
            f.write(edges)

    def get_adj(self):
        n=self.word_vectors.shape[0]
        edge_index = torch.tensor(list(self.edges_weight.keys())).t()
        adj = sp.coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
                            shape=(n, n), dtype='float32')
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        self.adj = adj

    def update(self, obs_list,name_list):
        """
        update graph by history observations
        update edges ,get edge pairs in rollouts.

        :param rollouts:
        :return:
        """
        # print("new_graph")

        update_edge_num = 0
        for ids in obs_list:
            for i in range(len(ids)):
                # for i in range(len(ids)):
                    if len(ids)>i+1:
                        # print(ids[i],ids[i+1])
                        if ids[i] in name_list and ids[i+1] in name_list:
                            nodes_1 = name_list[ids[i]]
                            nodes_2 = name_list[ids[i+1]]
                            if nodes_1 < nodes_2:
                                edge = (nodes_1, nodes_2)
                            else:
                                edge = (nodes_2, nodes_1)
                                if edge not in self.edges_weight:
                                    # print(edge)
                                    self.edges_weight[edge] = 0
                                else:
                                    self.edges_weight[edge] += 1

