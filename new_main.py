# import sys
# import os
# import json
import torch
from arguments import get_args
from algo.model import Policy
from algo.a2c_acktr import A2C_ACKTR
from utils import *
from train import train
from test import test,random_test
import time

ALL_ROOMS = {
    0: "Kitchens",
    1: "Living Rooms",
    2: "Bedrooms",
    3: "Bathrooms",
    4  :"all_uu",
    5:"all_ss",
    6:"all_su",
    7:"all_us",
    8:"k_2"

}


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda:2' if args.cuda else 'cpu')
    config = read_config(args.config_file)


    if args.history_folder and args.test != 2 :
        print("train or test by the from the history,initialize by the folder", args.history_folder)
        class DictToStruct:
            def __init__(self,**entries):
                self.__dict__.update(entries)

        weights, arguments,  graph = read_weights(args.history_folder,device)
        actor_critic, agent, rollouts = build_class_objecs(DictToStruct(**arguments), config, device)
        actor_critic.load_state_dict(torch.load(weights, map_location=device))




    else:
        # arguments = vars(args)
        graph = BasicGraph()
        graph.read_edge_from_json("/home/user/pythonwork/navigation/graph/edge.pkl")
        graph.read_node_from_json("/home/user/pythonwork/navigation/graph/node.pkl")
        graph.word_vectors.to(device)
        if args.norm_gcn:
            graph.get_adj()



    if args.test:
        if args.test !=2:
            print("test from file", args.history_folder)
            assert "actor_critic" in vars().keys()
            print(list(zip(range(len(ALL_ROOMS)), list(ALL_ROOMS.values()))))
            command = input("Please specify test  type:")
            test_type = ALL_ROOMS[int(command)]
            test(test_type, args, config, graph, device, actor_critic, agent,rollouts)
        elif args.test ==2:
            print("random_test")

            print(list(zip(range(len(ALL_ROOMS)), list(ALL_ROOMS.values()))))
            command = input("Please specify test  type:")
            test_type = ALL_ROOMS[int(command)]
            random_test(test_type, args, config, graph , device  )


    else:
        print("train start")
        print(list(zip(range(len(ALL_ROOMS)), list(ALL_ROOMS.values()))))
        command = input("Please specify room type:")
        scene_type = ALL_ROOMS[int(command)]
        args.run_dir = args.run_dir +scene_type+"/"+time.strftime('%m_%d_%H_%M',time.localtime(time.time()))

        if  "actor_critic" in vars().keys():
            train(scene_type, arguments, config, graph,device,actor_critic,agent,rollouts)
        else:
            train(scene_type, args, config, graph, device)



if __name__=="__main__":
    main()


    