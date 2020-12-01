import torch
import torch.nn as nn
import torch.optim as optim
from .kfac import KFACOptimizer

class A2C_ACKTR():
    def __init__(self, actor_critic, value_loss_coef, entropy_coef,
                 lr=None, eps=None, alpha=None, max_grad_norm=None, acktr=False):
        self.actor_critic = actor_critic
        self.acktr = acktr
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, graph, rollouts):
        # obs_shape = rollouts.obs[0].shape
        feature_shape = rollouts.feature[0].shape
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        """rollouts.goal_idx[-1].detach(),
            rollouts.feature[-1],
            rollouts.masks[-1]"""

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(

            (rollouts.feature[:-1], rollouts.recurrent_hidden_states[0]),
            rollouts.obs[:-1],
            rollouts.actions.view(-1, action_shape),
            rollouts.goal_idx[-1],
            rollouts.masks[:-1]
        )
        if len(torch.where(rollouts.masks==0)[0])>0:
            min_done = torch.where(rollouts.masks==0)[0][0]
            # assert rollouts.returns[min_done]==10
        else:
            min_done = num_steps

        values = values.view(num_steps, num_processes, 1)[0:min_done,:,:]
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)[0:min_done,:,:]
        if len(torch.where(action_log_probs>0)[0])>0:
            raise NameError
        advantages = rollouts.returns[:-1][0:min_done,:,:] - values
        advantages = advantages[0:min_done,:,:]
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()
        loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * dist_entropy
        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()
            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()
            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        loss.backward()

        if not self.acktr:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()
        return value_loss.item(), action_loss.item(), dist_entropy.item()
