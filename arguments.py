import os
import argparse
import time


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--experiment', default='new',
                        help='baseline or ours method')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | acktr | ppo')

# --------------------------------------------------------------------------------
    parser.add_argument('--test', type=int, default=0,
                        help='whether to activate testing phase')
    parser.add_argument('--history_folder', type=str, default=None)
    parser.add_argument('--run_type', type=int, default=1,
                        help='0: train, 1:debug,2:plot_process_image ')

#---------------------------------------------------------
    #lstm
    parser.add_argument('--base_hidden_size', type=int, default=512)
    parser.add_argument("--gru_single_input",type = int,default = 0)
    parser.add_argument("--norm_gcn",type = int,default = 0)
    #gcn
    parser.add_argument('--gcn_output_size', type=int, default=300)
    parser.add_argument('--gcn_input_size', type=int, default=300)
    parser.add_argument('--use_gcn', type=int, default=1,
                        help='whether to include gcn')
    parser.add_argument('--only_gcn', type=int, default=0,
                        help='only gcn')
    parser.add_argument('--use_gru', type=int, default=0,
                        help='whether to use gru to extract visual feature')
    parser.add_argument('--use_graph_update', type=int, default=1,
                        help='whether to include graph update')
#--------------------------------------------------------------------
    # env
    parser.add_argument('--feature_shape', type=int, default=[2048])
    parser.add_argument('--angle', type=float, default=90,
                        help='rotation angle')




#---------------------------------------------------

    parser.add_argument('--distance_type', type=str, default="cosine",
                        help='number of epochs to run on each thread')
    parser.add_argument('--num_epochs', type=int, default=50000,
                        help='number of epochs to run on each thread')
    parser.add_argument('--max_episode_length', type=int, default=100,
                        help='maximum length of an episode (default: 1000)')


    parser.add_argument('--with-goal', default=False)
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    # parser.add_argument('--is-attention', action='store_true', default=True)
    # parser.add_argument('--num-attention-heads', type=int, default=8)
    # parser.add_argument('--attention-hidden-size', type=int, default=16)
    # parser.add_argument('--num-attention-iteration', type=int, default=1)
    parser.add_argument('--use_bias_mtx', type=bool, default=True)



    parser.add_argument('--use_base_net', type=int, default=0,
                        help='whether to re-train cnn module')
    parser.add_argument('--history_size', type=int, default=3,
                        help='whether to stack frames')
    parser.add_argument('--action_size', type=int, default=4,
                        help='number of possible actions')
    parser.add_argument('--hard', type=int, default=0,
                        help='whether to make environment harder\
                            0: agent only has to reach the correct position\
                            1: agent has to reach the correct position and has right rotation')

    parser.add_argument('--lr', type=float, default=2.5e-3,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--seed', type=int, default=1777,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=100,
                        help='number of forward steps in A2C (default: 20)')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')



# --------------------------------------------------------------------------------


#--------------------------------------------------------------------

    parser.add_argument('--anti_col', type=int, default=0,
                        help='whether to include collision penalty to rewarding scheme')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='ppo batch size (default: 64)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='log interval, one log per n updates (default: 100)')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=5e7,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--name', default='dm_small',
                        help='environment to train on (default: VizDoom)')
    parser.add_argument('--doom_wad', type=str,
                        default='data/train/deepmind_small.wad_manymaps_test.wad')
    parser.add_argument('--cuda', action='store_true', default=True
                        ,
                        help='disables CUDA training')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-vis', action='store_true', default=True,
                        help='disables visdom visual,; mlnkp09ization')
    parser.add_argument('--config_file', type=str, default="config.json")

    args = parser.parse_args()

    if args.recurrent_policy:
        args.run_dir = os.path.join('./runs/',
                                       '%s_%s_%s_seed_%d_%s_%s_%s_atiter_' % (args.env_name, args.experiment, args.algo,
                                                                args.seed, 'rgb' if args.is_rgb else 'gray',
                                                                             'goal' if args.with_goal else 'nogoal',
                                                                # args.num_attention_heads, args.attention_hidden_size,
                                                                'biasmtx' if args.use_bias_mtx else 'nobiaxmtx',
                                                                # args.num_attention_iteration
                                                                                                         ))
    else:
        args.run_dir = os.path.join('./runs/',
                                           '%s_%s_' % (args.name, args.experiment)
                                    )

    args.log_dir = os.path.join(args.run_dir, 'logs')
    args.feature_shape =[2048*args.action_size]
    if args.hard:
        args.action_size += 1#add the stop action


    return args
