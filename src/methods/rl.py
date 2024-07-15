"""
Classes for Reinforcement Learning
"""
import os
import random
import pickle
from collections import defaultdict, deque
from typing import NoReturn

from tqdm.auto import tqdm

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions as dist


# ##########
# CLASSES #
# ##########

class PPOPolicyNetwork(nn.Module):
    def __init__(self, in_dim, hid_dim, n_actions=2):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.relu1 = nn.ReLU6()
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.ReLU6()
        self.policy_head = nn.Linear(hid_dim, n_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        policy = self.softmax(self.policy_head(x))
        return policy


class PPOValueNetwork(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.relu1 = nn.ReLU6()
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.ReLU6()
        self.value_head = nn.Linear(hid_dim, 1)
        
    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        value = self.value_head(x)
        return value


class PPOAgent:
    def __init__(self, in_dim, hid_dim, lr, gamma, clip, save_dir, update_freq=4, bsz=128, device='cpu'):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.lr = lr
        self.gamma = gamma
        self.clip = clip
        self.update_freq = update_freq
        self.bsz = bsz
        self.save_dir = save_dir
        self.device = device

        self.policy_net = PPOPolicyNetwork(in_dim, hid_dim).to(device)
        self.value_net = PPOValueNetwork(in_dim, hid_dim).to(device)
        self.policy_opt = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=0.05)
        self.value_opt = optim.AdamW(self.value_net.parameters(), lr=lr, weight_decay=0.05)
        self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.policy_opt, T_max=1000)
        self.value_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.value_opt, T_max=1000)

        self.memory = deque(maxlen=10000)

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0).to(self.device)
        policy = self.policy_net(state)
        policy_dist = dist.Categorical(policy)
        action = policy_dist.sample()
        log_p = policy_dist.log_prob(action)
        return action.item(), log_p.item()

    def generalized_advantage_estimator(self, rewards, values, next_values, dones):
        td_deltas = [
                r + self.gamma * (1-done)*next_val - val for r, val, next_val, done in zip(
                    rewards, values, next_values, dones
                    )
                ]
        gae = 0
        advantages = []
        for delta in reversed(td_deltas):
            gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0,gae)
        return advantages

    def learn(self):
        if len(self.memory) < self.bsz:
            return False
        else:
            k = self.bsz

        self.policy_net.train()
        self.value_net.train()

        sample = random.sample(self.memory, k=k)
        states, actions, rewards, next_states, dones, log_ps = zip(*sample)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.long).to(self.device)
        log_ps = torch.tensor(np.array(log_ps), dtype=torch.float32).to(self.device)

        values = self.value_net(states).squeeze().detach()
        next_values = torch.cat((values[1:], torch.tensor([0]).to(self.device))).to(self.device)
        advantages = self.generalized_advantage_estimator(rewards, values, next_values, dones)
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(
            [adv + val for adv, val in zip(advantages, values)],
            dtype=torch.float32
        ).to(self.device)

        for _ in range(self.update_freq):
            policy = self.policy_net(states)
            policy_dist = dist.Categorical(policy)
            new_log_ps = policy_dist.log_prob(actions).to(self.device)
            entropy = policy_dist.entropy()

            values = self.value_net(states).squeeze()
            ratio = torch.exp(new_log_ps - log_ps)
            sur1 = ratio * advantages
            sur2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) *advantages

            policy_loss = -torch.min(sur1, sur2).mean() -0.01*entropy.mean()
            value_loss = F.mse_loss(returns, values)

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

        return True

    def train(self, epochs: int, train_env, valid_env):
        epoch_actions = {}
        epoch_rewards = {}
        best_rewards = -1e10
        epoch_since_improvement = 0
        best_epoch = None

        for epoch in tqdm(range(epochs), desc='Epochs'):
            pbar = tqdm(total=train_env.total_t)

            # Reset the environment for each epoch
            state = train_env.reset()
            next_state = None
            done = False

            actions = {
                dt: [] for dt in train_env.dates
            }
            daily_rewards = {
                dt: [] for dt in train_env.dates
            }

            while not done:
                # Select an action per the policy
                action, log_p = self.act(state)
                # Act on environment
                next_state, reward, done, day_done = train_env.step(action)
                # Store in replay memory
                self.memory.append(
                    (state, action, reward, next_state, done, log_p)
                )
                # Advance to next state
                state = next_state
                actions[train_env.current_date].append(action)
                daily_rewards[train_env.current_date].append(reward)
                pbar.update(1)

                if day_done or done or (train_env.current_time % 4 == 0):
                    learned = self.learn()

            if learned:
                self.policy_scheduler.step()
                self.value_scheduler.step()
                
            epoch_actions.update(
                {epoch: actions}
            )
            epoch_rewards.update(
                {epoch: daily_rewards}
            )

            # Eval
            _, valid_rewards = self.run(valid_env)
            valid_mean = np.array([r for dr in valid_rewards.values() for r in dr]).mean()
            if valid_mean > best_rewards:
                print("Better mean reward! Saving...")
                best_rewards = valid_mean
                torch.save(
                    self.policy_net.state_dict(),
                    os.path.join(self.save_dir, 'ppo-policy-network.pt')
                )
                pickle.dump(
                    self.value_net.state_dict(),
                    open(os.path.join(self.save_dir, 'ppo-value-network.pkl'),'wb')
                )
                best_epoch = epoch
                epoch_since_improvement = 0
            else:
                epoch_since_improvement += 1
                if epoch_since_improvement == -1:
                    break
            pbar.set_postfix(
                {
                    'Mean Daily Train Reward': np.array([r for dr in daily_rewards.values() for r in dr]).mean(),
                    'Best Mean Daily Valid Reward': best_rewards,
                    'LR': self.policy_scheduler.get_last_lr()
                }
            )
        print(best_epoch, best_reward)
        return actions, daily_rewards

    def run(self, env):
        self.policy_net.eval()
        self.value_net.eval()
        state = env.reset()
        next_state = None
        done = False
        reward = 0.0
        actions = {
            dt: [] for dt in env.dates
        }
        daily_rewards = {
            dt: [] for dt in env.dates
        }

        while not done:
            daily_rewards[env.current_date].append(reward)
            action, _ = self.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            actions[env.current_date].append(action)

        return actions, daily_rewards

    def to(self, device):
        self.device = device
        self.policy_net.to(device)
        self.value_net.to(device)



class MCTSNetwork(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, n_actions: int=2, device: str='cpu'):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.relu1 = nn.ReLU6()
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.ReLU6()

        # Policy
        self.policy_head = nn.Linear(hid_dim, n_actions)
        self.softmax = nn.Softmax(dim=1)

        # Value
        self.value_head = nn.Linear(hid_dim, 1)

        self.device = device

    def forward(self, state):
        state = state.to(self.device)
        x = self.relu1(self.linear1(state))
        x = self.relu2(self.linear2(x))
        policy = self.softmax(self.policy_head(x))
        value = self.value_head(x)

        return policy, value


class MCTSRecurrentNetwork(nn.Module):
    def __init__(self, in_dim, hid_dim, n_actions=2, device='cpu'):
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layes=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        self.relu1 = nn.ReLU()
        self.linear = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.ReLU()

        # Policy
        self.policy_head = nn.Linear(hid_dim, n_actions)
        self.softmax = nn.Softmax(dim=1)

        # Value
        self.value_head = nn.Linear(hid_dim,1)
        self.tanh = nn.Tanh()

        self.device = device

    def forward(self, state):
        x = self.relu1(self.lstm(state))
        x = self.relu2(self.linear(state))
        policy = self.softmax(self.policy_head(x))
        value = self.tanh(self.value_head(x))

        return policy, value


class Node:
    def __init__(self, state, parent=None, prior=0.0, depth:int=0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value = 0
        self.prior = prior

    def calculate_value(self):
        if self.visit_count == 0:
            return 0
        return self.value / self.visit_count

    def upper_confidence_bound_for_trees(self, uct):
        if self.visit_count == 0:
            return np.inf
        new_value = self.calculate_value() + (uct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        self.prior = new_value
        return new_value


class MonteCarloTreeSearch:
    def __init__(
        self,
        neural_net: nn.Module,
        uct: float=1.0,
        n_sim: int=100,
        n_actions: int=2
    ):
        self.net = neural_net
        self.uct = uct
        self.n_sim = n_sim
        self.n_actions = n_actions

    def select(self, node: Node) -> Node:
        best_score = -np.inf
        best_child = None
        for child in node.children.values():
            child_score = child.upper_confidence_bound_for_trees(
                self.uct
            )
            if child_score > best_score:
                best_score = child_score
                best_child = child
        return best_child

    def expand_and_evaluate(self, node: Node, env) -> float:
        state = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
        policy, value = self.net(state)
        policy = policy.squeeze().cpu().detach().numpy()
        value = value.item()
        for action in range(self.n_actions):
            if action not in node.children:
                next_state = self.transition_state(action, env)
                node.children.update(
                    {action: Node(next_state, node, policy[action])}
                )
        return value

    def transition_state(self, action: int, env) -> torch.Tensor:
        next_state = env.forgetful_state_transition(action)
        return next_state

    def backprop(self, path: list[Node], value: float) -> NoReturn:
        for node in reversed(path):
            node.visit_count += 1
            node.value += value

    def simulate(self, root: Node, env):
        for _ in range(self.n_sim):
            node = root
            path = [node]

            while node.children:
                node = self.select(node)
                path.append(node)

            value = self.expand_and_evaluate(node, env)

            self.backprop(path, value)

        return max(root.children.items(), key=lambda child: child[1].visit_count)[0]


class NeuralNetGuidedMCTS:
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        save_dir: str,
        n_actions: int=2,
        n_sim: int=100,
        lr: float=1e-3,
        weight_decay: float=0.1,
        gamma: float=0.999,
        bsz: int=32,
        device: str='cpu'
    ):
        self.net = MCTSNetwork(
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_actions=n_actions,
            device=device
        ).to(device)
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=lr
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            eta_min=1e-6,
            T_max = 100,
            last_epoch=-1
        )
        self.mcts = MonteCarloTreeSearch(
            self.net,
            n_sim=n_sim,
            n_actions=n_actions
        )
        self.save_dir = save_dir
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.bsz = bsz
        self.device = device

    def learn(self):
        if len(self.memory) < self.bsz:
            k = len(self.memory)
        else:
            k = self.bsz

        self.net.train()
        self.optimizer.zero_grad()

        sample = random.sample(self.memory, k=k)
        states, actions, rewards, next_states, dones = zip(*sample)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.long).to(self.device)

        policies, values = self.net(states)
        values = values.squeeze()
        discounted_reward = (
            rewards + self.gamma * self.net(next_states)[1].squeeze() * (1 - dones) 
        )

        loss = F.mse_loss(values, discounted_reward) \
                + F.cross_entropy(policies, actions)

        nn.utils.clip_grad_norm_(self.net.parameters(), 1.5)
        loss.backward()
        self.optimizer.step()
        self.net.eval()
        return True

    def train(self, epochs: int, train_env, valid_env):
        epoch_actions = {}
        epoch_rewards = {}
        best_rewards = -1e10
        epoch_since_improvement = 0
        best_epoch = None

        for epoch in tqdm(range(epochs), desc='Epochs'):
            pbar = tqdm(total=train_env.total_t)

            # Reset the environment for each epoch
            state = train_env.reset()
            next_state = None
            done = False

            actions = {
                dt: [] for dt in train_env.dates
            }
            daily_rewards = {
                dt: [] for dt in train_env.dates
            }

            while not done:
                # Create a new node for the current state
                node = Node(state)
                # Select an action per the MCTS
                action = self.mcts.simulate(node, train_env)
                # Act on environment
                next_state, reward, done, day_done = train_env.step(action)
                # Store in replay memory
                self.memory.append(
                    (state, action, reward, next_state, done)
                )
                # Advance to next state
                state = next_state
                actions[train_env.current_date].append(action)
                daily_rewards[train_env.current_date].append(reward)
                pbar.update(1)

                if day_done or done or (train_env.current_time % 4 == 0):
                    learned = self.learn()
                
            if learned:
                self.scheduler.step()
            
            
            epoch_actions.update(
                {epoch: actions}
            )
            epoch_rewards.update(
                {epoch: daily_rewards}
            )

            # Eval
            _, valid_rewards = self.run(valid_env)
            valid_mean = np.array([r for dr in valid_rewards.values() for r in dr]).mean()
            if valid_mean > best_rewards:
                print("Better mean reward! Saving...")
                best_rewards = valid_mean
                torch.save(
                    self.net.state_dict(),
                    os.path.join(self.save_dir, 'network.pt')
                )
                pickle.dump(
                    self.mcts,
                    open(os.path.join(self.save_dir, 'mcts.pkl'),'wb')
                )
                best_epoch = epoch
                epoch_since_improvement = 0
            else:
                epoch_since_improvement += 1
                if epoch_since_improvement == -1:
                    break
            pbar.set_postfix(
                {
                    'Mean Daily Train Reward': np.array([r for dr in daily_rewards.values() for r in dr]).mean(),
                    'Best Mean Daily Valid Reward': best_rewards,
                    'LR': self.scheduler.get_last_lr()
                }
            )
        print(best_epoch, best_reward)
        return actions, daily_rewards

    def run(self, env):
        state = env.reset()
        next_state = None
        done = False
        reward = 0.0
        actions = {
            dt: [] for dt in env.dates
        }
        daily_rewards = {
            dt: [] for dt in env.dates
        }

        while not done:
            daily_rewards[env.current_date].append(reward)
            node = Node(state)
            action = self.mcts.simulate(node, env)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            actions[env.current_date].append(action)

        return actions, daily_rewards

    def to(self, device):
        self.device = device
        self.net.to(device)
