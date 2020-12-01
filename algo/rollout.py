import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage():
    def __init__(self, num_steps, num_processes, imgs_shape, action_space, recurrent_hidden_state_size=1):
        self.obs = []
        self.obs_states = {}
        # self.feature = []
        self.feature = torch.zeros(num_steps + 1, num_processes, *imgs_shape)
        self.curr_idx = torch.zeros(num_steps + 1, num_processes, 1, dtype=torch.int32)
        self.goal_idx = torch.zeros(num_steps + 1, num_processes, 1, dtype=torch.int32)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        action_shape = 1

        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.actions = self.actions.long()
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
#        self.obs = self.obs.to(device)
        self.curr_idx = self.curr_idx.to(device)
        self.goal_idx = self.goal_idx.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs_states,objs_index, image_features,  goal_idx, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards,masks):
        """
        current_obs, objs_index,image_features,goal_obj,
        recurrent_hidden_states, action, action_log_prob, value, reward
        """
        self.obs.append(objs_index)
        # self.obs_states[self.step + 1] = obs_states
        # self.feature.append(image_features)
        self.feature[self.step + 1].copy_(torch.from_numpy(image_features))
        # self.curr_idx[self.step + 1].copy_(torch.from_numpy(np.asarray(curr_idx, dtype=np.int64)).unsqueeze(-1))
        self.goal_idx[self.step + 1].copy_(torch.from_numpy(np.asarray([goal_idx], dtype=np.int64)))
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states.squeeze())
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(torch.from_numpy(np.asarray([rewards], dtype=np.float)))
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """
        rollouts.goal_idx[-1].item(),
        (rollouts.feature[-1],rollouts.recurrent_hidden_states[-1])
        rollouts.masks[-1]
        :return:

        """
        # self.obs[0].copy_(self.obs[-1])
        # self.curr_idx[0].copy_(self.curr_idx[-1])
        obs = self.obs[-1]
        self.obs = []
        self.obs.append(obs)
        self.goal_idx[0].copy_(self.goal_idx[-1])
        self.feature[-1].copy_(self.feature[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        # self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            if len(torch.where(self.masks == 0)[0]) > 0:
                min_done = torch.where(self.masks == 0)[0][0]
                # assert rollouts.returns[min_done]==10
            else:
                min_done = self.num_steps
            for idx in reversed(range(min_done)):
                delta = self.rewards[idx] + gamma * self.value_preds[idx+1]  - self.value_preds[idx]
                gae = delta + gamma * tau * gae
                self.returns[idx] = gae + self.value_preds[idx]
                # if self.returns[idx]>1:
                #     pass
        else:
            self.returns[-1] = next_value
            for idx in reversed(range(self.rewards.size(0))):
                self.returns[idx] = self.rewards[idx] + gamma * self.returns[idx+1] * self.masks[idx+1]

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                                                                                   self.recurrent_hidden_states.size(
                                                                                   -1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_idx in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                idx = perm[start_idx + offset]
                obs_batch.append(self.obs[:-1, idx])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, idx])
                actions_batch.append(self.actions[:, idx])
                value_preds_batch.append(self.value_preds[:-1, idx])
                return_batch.append(self.returns[:-1, idx])
                masks_batch.append(self.masks[:-1, idx])
                old_action_log_probs_batch.append(self.action_log_probs[:, idx])
                adv_targ.append(advantages[:, idx])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)
            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ
