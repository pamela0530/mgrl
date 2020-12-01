import matplotlib.pyplot as plt
import graph
import networkx as nx
import os
import pickle
from utils import *
from arguments import *

file_1 = "/home/user/pythonwork/navigation/graph/edge.pkl"
file_2 = "//home/user/pythonwork/navigation/runs/train-no_grubase_net_baseline_Kitchens/09_25_04_35/Kitchens.pkl"
file_3 = "/home/user/pythonwork/navigation/runs/train-our_update_baseline_all/09_23_23_49/all.pkl"
file_4 = "/home/user/pythonwork/navigation/runs/train-no_gru_new_all_uu/11_15_21_33/all_uu.pkl"
file_5 = "/home/user/pythonwork/navigation/runs/train-our_update_baseline_Kitchens/09_23_22_49/Kitchens.pkl"
file_6 = "/home/user/pythonwork/navigation/runs/train-our_new_all_uu/11_15_21_33/all_uu.pkl"
file_7 = "/home/user/pythonwork/navigation/runs/train-our_update_baseline_Bedrooms/09_23_22_52/Bedrooms.pkl"
scene = "FloorPlan19"
args = get_args()
torch.manual_seed(args.seed)

device = torch.device('cuda:3' if args.cuda else 'cpu')
config = read_config(args.config_file)
nodes_index = config["new_objects"]
def plot_graph(new_graph_file):
    config = read_config("/home/user/pythonwork/navigation/config.json")
    nodes_list = config["new_objects"]
    nodes_dic = {value: key for key, value in nodes_list.items()}

    print(nodes_dic)
    n=10
    m=10
    init_graph = graph.BasicGraph()
    init_graph.read_edge_from_json("/home/user/pythonwork/navigation/graph/edge.pkl")
    init_graph.read_node_from_json("/home/user/pythonwork/navigation/graph/node.pkl")
    graph_2 = graph.BasicGraph()
    f = open(new_graph_file, "rb")
    graph_2.edges_weight = pickle.load(f)["graph"]
    f.close()
    edge_1 = []
    edges_2 = []
    print(len(graph_2.edges_weight))
    print(len(init_graph.edges_weight))
    edgecolor = ["g" for i in range(len(init_graph.edges_weight.keys()))]
    edgecolor = edgecolor[:n]
    m_i=0
    for edge in graph_2.edges_weight.keys():
        if edge not in init_graph.edges_weight.keys():
            edges_2.append(edge)
            edgecolor.append("r")
            m_i+=1
            if m_i>=m:
                break
            print(edge)

    e_list = list(init_graph.edges_weight.keys())[:n] + edges_2[:m]
    G = nx.Graph()
    nodes_dic_1 = {}
    for edge in e_list:
        if edge[0] != edge[1]:
            if edge[0] not in nodes_dic_1:
                nodes_dic_1[edge[0]] = nodes_dic[edge[0]]
            if edge[1] not in nodes_dic_1:
                nodes_dic_1[edge[1]] = nodes_dic[edge[1]]

                # for i in range(104):
    # G.add_node(i)

    G.add_edges_from(e_list)
    nx.draw_networkx(G, pos=nx.random_layout(G), edgelist=e_list, edge_color=edgecolor,
                     with_labels=True, width=0.5,
                     alpha=0.9, node_size=10, labels=nodes_dic_1, node_color="y", font_size=10)
    plt.show()

def get_distance(args, config, scene,obj_1,obj_2):
    envs = MultiSceneEnv(scene, config, args)
    pass
if  __name__ == "__main__":
    plot_graph(file_7)

    # for i in range(104):
    #     obj_1,id_1 = nodes_index.item()[i]
    #     obj_2,id_2 =nodes_index.item()[i+1]
    #
    #     get_distance(args, config, scene,obj_1,obj_2)





