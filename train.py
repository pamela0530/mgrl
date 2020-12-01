
from utils import *
import random
import torch
import numpy as np
import torch.nn.functional as F
import time
import copy
import pickle
from tensorboardX import SummaryWriter


def train(scene_type, args, config, graph = None, device = None, policy = None, agent = None):
    # if scene_type != "all":

    scene_list = config['rooms'][scene_type]['train_scenes']
    # else:
    #     scene_list = []
    #     for type_i in

    print("training_sence:", (len(scene_list)), scene_list)
    if args.run_type ==0:
        writer = SummaryWriter(args.run_dir)
    if policy is None:
        actor_critic, agent, rollouts = build_class_objecs(args,config,device)

    print(actor_critic)
    value_loss_history = []
    action_loss_history = []
    dist_entropy_history = []
    total_loss_history = []
    reward_list = []
    success = []
    obs_list = []


    for j in range(args.num_epochs):
        # if j%args.log_interval==0:
        #     success = torch.zeros((args.log_interval))
        scene = random.choice(scene_list)
        start = time.time()
        envs = MultiSceneEnv(scene, config, args)
        state, score, target = envs.reset()
        # goal_obj = actor_critic.categories.index(target)
        goal_obj = config["new_objects"][target]
        rollouts.goal_idx[0].copy_(torch.from_numpy(np.asarray([goal_obj], dtype=np.int64)))
        rollouts.obs =[]
        rollouts.obs.append(score)
        rollouts.feature[0].copy_(torch.tensor(state))
        hx = torch.zeros(args.num_processes, args.base_hidden_size).to(device)
        rollouts.recurrent_hidden_states[0].copy_(hx)
        hx = hx.unsqueeze(0).to(device)
        episode_rewards = torch.zeros([args.num_processes, 1])
        final_rewards = torch.zeros([args.num_processes, 1])
        success_i = []
        for step in range(args.num_steps):
            with torch.no_grad():

                # graph.update(rollouts)


                value, action, action_log_prob, hx = actor_critic.act((rollouts.feature[step], hx),
                                                                          score, target,graph)

            # print(action)
            if len(torch.where(action_log_prob > 0)[0]) > 0:
                raise NameError
            state,score,reward, done = envs.step(action)
            if args.use_graph_update:


                obs_list.append(score[0])
            if done:
                success_i.append(1.0)
            else:
                success_i.append(0.0)
                # print("------------------------------------------------sucess in episode,", j)
            episode_rewards += reward
            raw_masks = torch.FloatTensor([[0.0] if done == 1 else [1.0]])
            # keeps the final_rewards always zero before done, and clean episode_reward when done.
            final_rewards *= raw_masks

            final_rewards += (1 - raw_masks) * episode_rewards
            episode_rewards *= raw_masks
            rollouts.insert(state, score, state, goal_obj,
                                hx, action, action_log_prob, value.squeeze(), reward,raw_masks)
        reward_list.append(final_rewards)
        if sum(success_i) >0:
            success.append(1.0)
        else:
            success.append(0.0)
        with torch.no_grad():
            '''graph,  index_list, goal_idx, feature'''
            next_value = actor_critic.get_value((rollouts.feature[-1], hx), score, target,graph).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(graph, rollouts)
        value_loss_history.append(value_loss)
        action_loss_history.append(action_loss)
        dist_entropy_history.append(dist_entropy)
        total_loss_history.append(
            action_loss + args.value_loss_coef * value_loss + args.entropy_coef * dist_entropy)
        rollouts.after_update()
        if args.use_graph_update:
            graph.update(obs_list, config["new_objects"])
            if args.norm_gcn:
                graph.get_adj()

        ##logs
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if (j % args.save_interval == 0 or j == args.num_epochs - 1):
            # save_model = actor_critic
            # if args.cuda:
            #     # actor_critic.attention_to_cpu()
            #     save_model = copy.copy(actor_critic).cpu()
            if not os.path.exists(args.run_dir):
                os.mkdir(args.run_dir)
            torch.save(actor_critic.state_dict(), os.path.join(args.run_dir, 'model_%d.pth' % total_num_steps))
        if j<=args.log_interval:
            success_rate = np.mean(success)
            mean_reward = np.mean(reward_list)
            min_reward = np.min(reward_list)
            max_reward = np.max(reward_list)
            median_reward = np.median(reward_list)

        else:
            success_rate = np.mean(success[-args.log_interval:])
            mean_reward = np.mean(reward_list[-args.log_interval:])
            min_reward = np.min(reward_list[-args.log_interval:])
            max_reward = np.max(reward_list[-args.log_interval:])
            median_reward = np.median(reward_list[-args.log_interval:])
        writer.add_scalar("scalar/scalar_success_rate", success_rate, j)
        writer.add_scalar("scalar/mean_reward", mean_reward, j)
        writer.add_scalar("scalar/value_loss", value_loss, j)
        writer.add_scalar("scalar/action_loss", action_loss, j)
        writer.add_scalar("scalar/entropy", dist_entropy, j)
        writer.add_scalar("scalar/median_reward", median_reward, j)
        writer.add_scalar("scalar/min_reward", min_reward, j)
        writer.add_scalar("scalar/loss", value_loss+action_loss, j)

        # Logging
        if j % args.log_interval == 0:
            # envs.log()
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} | "
                  "reward: mean/median {:.1f}/{:.1f}, min/max {:.1f}/{:.1f} | "
                  "entropy {:.3f}, value_loss {:.3f}, action_loss {:.3f},success_rate {:.3f}".
                  format(j, total_num_steps,
                         int(total_num_steps / (end - start)),
                         mean_reward,
                         median_reward,
                         min_reward,
                         max_reward, dist_entropy,
                         value_loss, action_loss,success_rate))
            with open('{}/{}.pkl'.format(args.run_dir, scene_type), 'ab') as f:
                pickle.dump({"updates" : j,
                             "reward":       reward_list[-1],
                             "mean_reward":  mean_reward,
                             "entropy":      dist_entropy,
                             "value_loss":    value_loss,
                             "action_loss": action_loss,
                            "success_rate":  success_rate,
                             "graph": graph.edges_weight,
                             },
                              f, pickle.HIGHEST_PROTOCOL)

        if j == 2:

            # with open('{}/{}.json'.format(args.run_dir, "config"), 'w') as f:
            #     json_data = json.dumps( config)
            #     f.write(json_data)
            with open('{}/{}.json'.format(args.run_dir, "arguments"), 'w') as f:
                json_data =    json.dumps(vars(args))
                f.write(json_data)

    time.sleep(5)
    try:
        os.system('cp train.log %s/' % args.experiment_dir)
    except:
        print('Copy train.log failed.')
        pass
    print('Training done.')
    writer.close()








