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


###########
# CLASSES #
###########

class MCTSNetwork(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, n_actions: int=2, device: str='cpu'):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.ReLU()

        # Policy
        self.policy_head = nn.Linear(hid_dim, n_actions)
        self.softmax = nn.Softmax(dim=1)

        # Value
        self.value_head = nn.Linear(hid_dim, 1)
        self.tanh = nn.Tanh()

        self.device = device

    def forward(self, state):
        state = state.to(self.device)
        x = self.relu1(self.linear1(state))
        x = self.relu2(self.linear2(x))
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
            return None

        self.net.train()
        self.optimizer.zero_grad()

        sample = random.sample(self.memory, k=self.bsz)
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

        loss.backward()
        self.optimizer.step()
        self.net.eval()

    def train(self, epochs: int, train_env, valid_env):
        epoch_actions = {}
        epoch_rewards = {}
        best_rewards = -1e10
        epoch_since_improvement = 0

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

                if day_done or done:
                    self.learn()
            
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
                    self.net,
                    os.path.join(self.save_dir, 'network.pt')
                )
                pickle.dump(
                    self.mcts,
                    open(os.path.join(self.save_dir, 'mcts.pkl'),'wb')
                )
                epoch_since_improvement = 0
            else:
                epoch_since_improvement += 1
                if epoch_since_improvement == 5:
                    break
            pbar.set_postfix(
                {
                    'Mean Daily Train Reward': np.array([r for dr in daily_rewards.values() for r in dr]).mean(),
                    'Best Mean Daily Valid Reward': best_rewards 
                }
            )
        return actions, daily_rewards

    def run(self, env):
        state = env.reset()
        next_state = None
        done = False
        reward = 0.01
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
