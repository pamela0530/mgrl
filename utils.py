import os
from graph import BasicGraph
import json
from algo.model import Policy
from algo.a2c_acktr import A2C_ACKTR
from ai2thor_envs.multi_objs_env import MultiSceneEnv
from algo.rollout import RolloutStorage
import torch
import pickle


def read_weights(folder,device):

        weights = [f for f in os.listdir(folder) if f.endswith('.pth')]
        histories = [f for f in os.listdir(folder) if f.endswith('.pkl')]
        history = histories[0]
        env_graph = BasicGraph()
        env_graph.read_node_from_json("/home/user/pythonwork/navigation/graph/node.pkl")
        f = open(os.path.join(folder,history), "rb")
        env_graph.edges_weight = pickle.load(f)["graph"]
        f.close()

        arguments = json.load(open(folder + '/arguments.json'))
        print(list(zip(range(len(weights)), weights)))
        wid = input("Please specify weights: ")
        weights = weights[int(wid)]
        # env_graph.word_vectors.to(device)
        return os.path.join(folder, weights), arguments, env_graph

def read_config(config_path):
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return config

def build_class_objecs(args, config,device):
    actor_critic = Policy(config,args,device)
    agent = A2C_ACKTR(actor_critic, args.value_loss_coef, args.entropy_coef,
                      lr=args.lr, eps=args.eps, alpha=args.alpha, max_grad_norm=args.max_grad_norm)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              args.feature_shape, args.action_size,
                              args.base_hidden_size)
    actor_critic.to(device)
    rollouts.to(device)
    # if args.
    return actor_critic,agent,rollouts

FixedCategorical = torch.distributions.Categorical
old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)
log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)
