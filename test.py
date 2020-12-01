from utils import *
import random
import torch
import numpy as np
import torch.nn.functional as F
import time
import copy
import pickle
from tensorboardX import SummaryWriter
import random

def test(test_type, args, config, graph , device , actor_critic, agent, rollouts ):
    scene_list = config['test'][test_type] = config['rooms'][test_type]['test_scenes']
    # scene_list = config['test'][test_type]
    print("test_sence:", (len(scene_list)), scene_list)



    value_loss_history = []
    action_loss_history = []
    dist_entropy_history = []
    total_loss_history = []
    reward_list = []
    success = []
    obs_list = []
    print(actor_critic)
    # writer = SummaryWriter("runs/"+args.run_dir)
    for j in range(args.num_epochs):
        # if j%args.log_interval==0:
        #     success = torch.zeros((args.log_interval))
        scene = random.choice(scene_list)
        print("test:",scene)
        start = time.time()
        try:
            envs = MultiSceneEnv(scene, config, args)
        except:
            print(scene)
            j=j-1
            continue
        # print(j)
        state, score, target = envs.reset()
        goal_obj = actor_critic.categories.index(target)
        rollouts.goal_idx[0].copy_(torch.from_numpy(np.asarray([goal_obj], dtype=np.int64)))
        rollouts.obs = []
        rollouts.obs.append(score)
        rollouts.feature[0].copy_(torch.tensor(state))
        hx = torch.zeros(args.num_processes, args.base_hidden_size).to(device)
        rollouts.recurrent_hidden_states[0].copy_(hx)
        hx = hx.unsqueeze(0).to(device)
        episode_rewards = torch.zeros([args.num_processes, 1])
        final_rewards = torch.zeros([args.num_processes, 1])
        for step in range(args.num_steps):
            with torch.no_grad():

                # graph.update(rollouts)


                value, action, action_log_prob, hx = actor_critic.act((rollouts.feature[step], hx),
                                                                      score, target, graph)

            # print(action)
            if len(torch.where(action_log_prob > 0)[0]) > 0:
                raise NameError
            state, score, reward, done = envs.step(action)
            if args.use_graph_update:
                obs_list.append(score[0])
            if done :
                success.append(1.0)
                break
                # print("------------------------------------------------sucess in episode,", j)
            # episode_rewards += reward
            # raw_masks = torch.FloatTensor([[0.0] if done == 1 else [1.0]])
            # # keeps the final_rewards always zero before done, and clean episode_reward when done.
            # final_rewards *= raw_masks
            #
            # final_rewards += (1 - raw_masks) * episode_rewards
            # episode_rewards *= raw_masks
            # rollouts.insert(state, score, state, goal_obj,
            #                 hx, action, action_log_prob, value.squeeze(), reward, raw_masks)
        # reward_list.append(final_rewards)
        if len(success) == j:
            success.append(0.0)


        # if args.use_graph_update:
        #     graph.update(obs_list, config["new_objects"])
        #     if args.norm_gcn:
        #         graph.get_adj()

        ##logs
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        # if (j % args.save_interval == 0 or j == args.num_epochs - 1):
        #     # save_model = actor_critic
        #     # if args.cuda:
        #     #     # actor_critic.attention_to_cpu()
        #     #     save_model = copy.copy(actor_critic).cpu()
        #     if not os.path.exists(args.run_dir):
        #         os.mkdir(args.run_dir)
            # torch.save(actor_critic.state_dict(), os.path.join(args.run_dir, 'model_%d.pth' % total_num_steps))
        if j <= 500:
            success_rate = np.mean(success)


        else:
            success_rate = np.mean(success[-500:])

        # writer.add_scalar("scalar/scalar_success_rate", success_rate, j)


        # Logging
        if j % args.log_interval == 0:
            # envs.log()
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} | "
                  "success_rate {:.3f}".
                  format(j, total_num_steps,
                         int(total_num_steps / (end - start)),
                          success_rate))
            # with open('{}/{}.pkl'.format(args.run_dir,"test_"+test_type), 'wb') as f:
            #     pickle.dump({"updates": j,
            #
            #                  "success_rate": success_rate,
            #                  "graph": graph.edges_weight,
            #                  },
            #                 f, pickle.HIGHEST_PROTOCOL)

        # if j == 2:
        #     with open('{}/{}.json'.format(args.run_dir, "arguments"), 'w') as f:
        #         json_data =    json.dumps(vars(args))
        #         f.write(json_data)


    print("sucess_rate:", np.mean(success),success_rate)

    time.sleep(5)
    try:
        os.system('cp train.log %s/' % args.experiment_dir)
    except:
        print('Copy train.log failed.')
        pass
    print('Training done.')
    # writer.close()



def random_test(test_type, args, config, graph , device ):

    # scene_list = config['test'][test_type]
    scene_list = config['test'][test_type] = config['rooms'][test_type]['test_scenes']
    print("test_sence:", (len(scene_list)), scene_list)

    success = []
    obs_list = []
    # writer = SummaryWriter("runs/"+args.run_dir)
    for j in range(args.num_epochs):
        # if j%args.log_interval==0:
        #     success = torch.zeros((args.log_interval))
        scene = random.choice(scene_list)
        start = time.time()
        envs = MultiSceneEnv(scene, config, args)
        state, score, target = envs.reset()
        goal_obj = config["new_objects"][target]

        action_space = [0,1,2]

        for step in range(args.num_steps):
            with torch.no_grad():
                action = random.randrange(3)
            # print("action",action)
            state, score, reward, done = envs.step(action)


            if done and len(success) == j:
                success.append(1.0)
                # print("epsode:",j,"done" )
        if len(success) == j:
            success.append(0.0)
        ##logs
        assert len(success) == j+1
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if (j % args.save_interval == 0 or j == args.num_epochs - 1):

            if not os.path.exists(args.run_dir):
                os.mkdir(args.run_dir)
            # torch.save(actor_critic.state_dict(), os.path.join(args.run_dir, 'model_%d.pth' % total_num_steps))
        if j <= args.log_interval:
            success_rate = np.mean(success)


        else:
            success_rate = np.mean(success[-args.log_interval:])

        # writer.add_scalar("scalar/scalar_success_rate", success_rate, j)


        # Logging
        if j % args.log_interval == 0:
            # envs.log()
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} | "
                  "success_rate {:.3f}".
                  format(j, total_num_steps,
                         int(total_num_steps / (end - start)),
                          success_rate))
            print(len(success))
            with open('{}/{}.pkl'.format(args.run_dir,"test_"+test_type), 'wb') as f:
                pickle.dump({"updates": j,

                             "success_rate": success_rate,
                             "graph": graph.edges_weight,
                             },
                            f, pickle.HIGHEST_PROTOCOL)

        if j == 2:
            with open('{}/{}.json'.format(args.run_dir, "arguments"), 'w') as f:
                json_data =    json.dumps(vars(args))
                f.write(json_data)


    print("sucess_rate:", np.mean(success),success_rate)

    time.sleep(5)
    try:
        os.system('cp train.log %s/' % args.experiment_dir)
    except:
        print('Copy train.log failed.')
        pass
    print('Training done.')
    # writer.close()





