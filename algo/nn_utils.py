import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# Categorical
FixedCategorical = torch.distributions.Categorical
old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)
log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

def get_obj_state(obj_sum, obj_lists):
    """
    transform obj_lists to matrixs(obj_sum * obj_sum)
    :return:
    matrix
    """
    ones_matrix = np.zeros((len(obj_lists), obj_sum, obj_sum))
    for i in range(len(obj_lists)):
        for obj_id in obj_lists[i]:
            ones_matrix[i][obj_id][obj_id] = 1
    return ones_matrix

