import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nn_utils import init, get_obj_state,FixedCategorical,Categorical
from torch_geometric.nn import GCNConv
from .pygcn.models import GCN as normGCN


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,device):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid, cached=False, bias=False)
        self.gc2 = GCNConv(nhid, nhid, cached=False, bias=False)
        self.gc3 = GCNConv(nhid, nclass, cached=False, bias=False)
        self.dropout = dropout


    def forward(self, x, adj):
        # print(x.device,adj.device)

        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # print("GCN_start1")
        #x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        return x


class Policy(nn.Module):

    def __init__(self, config, arguments, device):
        super(Policy, self).__init__()
        self.config = config
        self.arguments = arguments
        self.device =device
        self.input_size = 2048
        self.hidden_size = 512
        self.gcn_output_size = 300
        self.categories = list(config['new_objects'].keys())
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        if self.arguments.hard:
            self.in_features = 2048*(self.arguments.action_size-1)
            self.action_size =self.arguments.action_size-1
        else:
            self.in_features = self.arguments.action_size*2048
            self.action_size = self.arguments.action_size
        if not arguments.use_gru:
            "visual feature use MLP to extract"
            self.visual_ft = nn.Linear(in_features=self.in_features, out_features=self.hidden_size)

            self.visual_size = self.hidden_size


        else:
            "visual feature use GRU to extract"
            if not arguments.gru_single_input:
                "GRU use multi images as input"

                self.visual_ft = nn.GRU(input_size=self.in_features, hidden_size=self.hidden_size)
                self.visual_size = self.hidden_size


            else:
                "single image as input"
                self.visual_ft = nn.GRU(input_size=2048 , hidden_size=self.hidden_size)
                self.visual_size =3*self.hidden_size


        if arguments.use_gcn:
            "use gcn "
            self.cate2idx = config['new_objects']
            self.num_objects = len(self.categories)

            # print("Create gcn!")
            if self.arguments.norm_gcn:
                self.gcn = normGCN(nfeat=300,
                           nhid=self.hidden_size,
                           nclass=self.gcn_output_size)
            else:
                self.gcn = GCN(nfeat=300,
                           nhid=self.hidden_size,
                           nclass=self.gcn_output_size,
                           dropout=True,
                           device=device)
            self.gcn.to(self.device)
            self.gcn_state_size = self.action_size
            if self.arguments.only_gcn :
                self.visual_size = 0

        else:
            "not use gcn. state only includes visual state"
            self.num_objects = 0
            self.global_state_size = self.hidden_size
            self.dist = Categorical(self.global_state_size , arguments.action_size)
            self.gcn_state_size = 0


        if self.arguments.use_base_net:
            "use linear mlp transform  the visual feature to 3_dim state. uneffective"
            self.base_nn = nn.Sequential(
                init_(nn.Linear(self.hidden_size, 3)), nn.ReLU()
            )
            self.visual_size=3

        self.global_state_size = self.visual_size+self.gcn_state_size
        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.dist = Categorical(self.global_state_size, arguments.action_size)
        self.critic_linear = nn.Linear(self.global_state_size, 1)
        self.actor_linear = nn.Linear(self.global_state_size, arguments.action_size)

        distance = {"cosine":nn.CosineEmbeddingLoss(margin=0,reduction = 'none'),
                    "kl":nn.KLDivLoss(reduce=False)

        }

        self.distance = distance[arguments.distance_type]

   # @property
    # def recurrent_hidden_state_size(self):zhineng
    #     return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_distance(self, new_state, goal_idx):
        (m,n,node_num,dim_n) = new_state.shape
        w_diss = torch.zeros(m,n)
        goal_feature = self.graph_nodes_features[goal_idx].unsqueeze(0)
        new_state.view(m*n, node_num,dim_n)
        for i in range(m):
            for j in range(n):
                node_i_feature = new_state[i][j]
                if self.arguments.distance_type == "cosine":
                    # dis_i = []
                    # for nodes_i in range(105):
                    #     dis_i.append(self.distance(node_i_feature[nodes_i].unsqueeze(0), goal_feature.unsqueeze(0),
                    #                                          torch.tensor(1.0).to(self.device)))
                    dis = self.distance(node_i_feature, goal_feature,
                                                             torch.tensor(1.0).to(self.device))
                    w_diss[i][j] = torch.min(dis,-1).values
                    # w_diss[i][j] = min(dis_i)
                else:
                    w_diss[i][j] = self.distance(node_i_feature.unsqueeze(0), goal_feature.unsqueeze(0))

        return w_diss

    def graph_initial(self, env_graph):
        nodes = env_graph.word_vectors.to(self.device)
        if  self.arguments.norm_gcn:
            edge_index = env_graph.adj.to(self.device)
        else:
            edge_index = torch.tensor(list(env_graph.edges_weight.keys())).t().to(self.device)
        self.graph_nodes_features = self.gcn(nodes, edge_index)
        # print(self.graph_nodes_features.shape)

    def visual_module(self, inputs):
        inputs, hx = inputs
        torch_inputs = inputs
        if not self.arguments.use_gru:
            joint_features = torch_inputs.view(1, -1).to(self.device)
            visual = F.relu(self.visual_ft(joint_features))
            visual = visual.unsqueeze(0)
        elif not self.arguments.gru_single_input:
            feature = torch_inputs.unsqueeze(0).to(self.device)
            _, hx = self.visual_ft(feature, hx )
            visual = hx
        else:
            visuals = []
            for i in range(3):
                input = inputs[:,i*2048:(i+1)*2048].unsqueeze(0).to(self.device)
                _, hx = self.visual_ft(input, hx)
                visuals.append(hx.squeeze())
            visual = torch.stack(visuals, -2).unsqueeze(0)
        if self.arguments.use_base_net:
            visual = self.base_nn(visual)
        else:
            hx =visual
        return visual, hx


    def act(self, inputs, scores, target, graph):
        x,hx = self.base_network(inputs, scores, target,graph)
        # if  self.arguments.only_gcn:
        #     value = torch.min(1-x)
        #     dist = self.dist(1-x)
        #     action = torch.argmin(x).view((1,1,1,1))
        #     action_log_probs = dist.log_probs(action)
        # else:
        value = self.critic_linear(x)
        dist = self.dist(x)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)
        if self.arguments.run_type ==2:
            print(action)
        return value.squeeze(), action.squeeze(), action_log_probs.squeeze(), hx

        # elif not self.arguments.use_gru:
        #     value = self.critic_linear(x)
        #     dist = self.dist(x)
        #     action = dist.sample()
        #     action_log_probs = dist.log_probs(action)
        #     return value.squeeze(), action.squeeze(), action_log_probs.squeeze(), hx
        # elif not self.arguments.gru_single_input:
        #     value = self.critic_linear(x)
        #     dist = self.dist(x)
        #     action = dist.sample()
        #     action_log_probs = dist.log_probs(action)
        #     return value.squeeze(), action.squeeze(), action_log_probs.squeeze(), hx
        # else:
        #     value_s = self.critic_linear(x)
        #     value = value_s.max()
        #     prob = value_s.softmax(1)
        #     dist = self.dist(torch.flatten(x))
        #     action = dist.sample()
        #     action_log_probs = dist.log_probs(action)
        #
        #     aa = action[0][0]
        #     if aa >=3:
        #         hx = inputs[-1]
        #     else:
        #         hx = hx[:,aa,:]
        #         hx = hx.unsqueeze(0)
        #     return value.squeeze(), action.squeeze(), action_log_probs.squeeze(), hx



    def get_index(self, objnames_list):
        index_list= []
        for objs in objnames_list:
            index_list.append([])
            for obj_name in objs:
                try:
                    index_list[-1].append(self.categories.index(obj_name))
                except:
                    pass
        return index_list


    def get_value(self, inputs, scores, target,graph):
        x,_= self.base_network(inputs, scores, target,graph)
        if not self.arguments.gru_single_input:
            return self.critic_linear(x)
        else:
            value_s = self.critic_linear(x)
            value = value_s.max()
            return value

    def base_network(self, inputs, scores, target,graph):
        if self.arguments.run_type == 1:
            print(scores,target)
        if self.arguments.use_gcn:
            self.graph_initial(graph)
            goal_idx = self.categories.index(target)
            scores_index = self.get_index(scores)
            obj_state = torch.from_numpy(get_obj_state(len(self.categories), scores_index)).unsqueeze(0).float().to(self.device)
            new_state = torch.matmul(obj_state, self.graph_nodes_features)
            # new_state = new_state.view(  self.num_objects * 3, self.gcn_output_size)
            w_diss = self.get_distance(new_state, goal_idx)
            if  self.arguments.gru_single_input:
                w_diss = w_diss.unsqueeze(-1)
            else:
                w_diss = w_diss.unsqueeze(1)
            if self.arguments.run_type == 2:
                print(scores, target,w_diss)

        x,hx = self.visual_module(inputs)
        if self.arguments.use_gcn:
            x = torch.cat((w_diss.to(self.device), x), -1)
        if self.arguments.only_gcn:
            x = w_diss.to(self.device)

        return x,hx


    def evaluate_actions(self, inputs, scores,action, target,mask):
        """
        graph, indexs_matrix, index_list, goal_idx, feature
        :param graph:
        :param index_list:
        :param goal_idx:
        :param feature:
        :param action:
        :return:
        """
        num_process = inputs[0].shape[1]
        step_size = inputs[0].shape[0]
        if self.arguments.use_gcn:
            goal_idx = target
            obj_states = []
            for score in scores:
                score_index=self.get_index(score)
                obj_states.append( torch.from_numpy(get_obj_state(len(self.categories), score_index)))
            obj_states = torch.stack(obj_states,dim=0).float().to(self.device)
            new_state = torch.matmul(obj_states, self.graph_nodes_features)
            w_diss = self.get_distance(new_state, goal_idx[0][0]).unsqueeze(1)
            # new_state = new_state.view(  step_size, 3, self.num_objects, self.gcn_output_size)
            # w_diss = torch.zeros(step_size,num_process,   3).to(self.device)
            #
            # for i in range(step_size):  # default = 1
            #     for j in range(num_process):
            #         for action_i in range(3):
            #             goal_feature = self.graph_nodes_features[goal_idx[i][j]].repeat(self.num_objects, 1)
            #             w_dis= self.distance(new_state[i][action_i], goal_feature,torch.tensor(1.0).to(self.device))
            #
            #             w_diss[i][j][action_i] = w_dis

        inputs, hx = inputs
        # torch_inputs = [torch.from_numpy(inp).type(self.dtype) for inp in inputs]
        if not self.arguments.use_gru:
            visual = F.relu(self.visual_ft(inputs.to(self.device)))
            # visual = visual.unsqueeze(0)
        elif not self.arguments.gru_single_input:
            hx = hx.unsqueeze(0)
            # feature = torch_inputs.unsqueeze(0).unsqueeze(0)
            # feature = torch_inputs.view(-1, self.input_size*3)
            hxs,_ = self.visual_ft(inputs.to(self.device), hx )
            visual = hxs
        else:
            inputs = inputs[:,:,0:2048]
            hx = hx.unsqueeze(0)
            # feature = torch_inputs.unsqueeze(0).unsqueeze(0)
            # feature = torch_inputs.view(-1, self.input_size*3)
            hxs, _ = self.visual_ft(inputs.to(self.device), hx)
            visual = hxs
            w_diss = w_diss[:,:,0:1]
        if self.arguments.use_base_net:
            visual=self.base_nn(visual)
        if self.arguments.use_gcn:
            if  self.arguments.only_gcn:
                x = w_diss.to(self.device)
            else:
                x = torch.cat((w_diss.to(self.device), visual), -1)
        else:
            x = visual
        # print("state", global_state.shape)
        # action = w_dis.argmin()
        # print(action)
        value = self.critic_linear(x)
        dist = self.dist(x)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, hx




