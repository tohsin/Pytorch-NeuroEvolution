import argparse
import copy
from functools import partial
import logging
import os
import pickle
import time
import sys
from SafeNeuroEvolution import SafeNeuroEvolution
from SafeNeuroEvolution  import SafeAgent
# from utils.helpers import weights_init
import safety_gymnasium
# import gym
# from gym import logger as gym_logger
import numpy as np
import torch
import torch.nn as nn

# gym_logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights_path', type=str, default="out.plt", help='Path to save final weights')
parser.add_argument('-c', '--cuda', action='store_true', help='Whether or not to use CUDA')
parser.set_defaults(cuda=False) 

args = parser.parse_args()

cuda = args.cuda and torch.cuda.is_available()

# add the model on top of the convolutional base
class Agent(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Agent, self).__init__()
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.model(x)

task_name  =  "SafetyPointGoal0-v0"
env = safety_gymnasium.make(task_name)
obs_size = env.observation_space.shape[0]
actions_space = 2
model = Agent(obs_size, [32, 32], output_size = actions_space)

if cuda:
    print("using Cuda")
    model = model.to('cuda')

def get_performance(safe_agent : SafeAgent, model, render=False):
    weights = safe_agent.weights
    with torch.no_grad():
        cloned_model = copy.deepcopy(model)
        for i, param in enumerate(cloned_model.parameters()):
            try:
                param.data.copy_(weights[i])
            except:
                param.data.copy_(weights[i].data)
        env = safety_gymnasium.make(task_name, render_mode="rgb_array")
        ob, _ = env.reset()
        #obs, reward, cost, terminated, truncated, info = env.step(act)
        total_reward = 0
        total_cost = 0
        while True:
            if render:
                env.render()
            batch = torch.from_numpy(ob[np.newaxis,...]).float()
            if cuda:
                batch = batch.cuda()

            prediction = cloned_model(batch)
            action = prediction.cpu().clone().data[0]
            # ob, reward, done, _ = env.step(action)
            ob, reward, cost, terminated, truncated, _ = env.step(action)

            total_reward += reward
            total_cost += cost
            if terminated or truncated:
                break

        env.close()
    safe_agent.set_performance(total_reward, total_cost)
    
    
partial_func = partial(get_performance, model = model)
mother_parameters = list(model.parameters())

# orginal

ne = SafeNeuroEvolution(
    mother_parameters, partial_func, population_size = 50,
    sigma=0.1, learning_rate=0.001, reward_goal=40, consecutive_goal_stopping=20,
    threadcount=50, cuda=cuda, render_test=False, task = task_name , select_random_parent = False
)

# Mac test code

# ne = SafeNeuroEvolution(
#     mother_parameters, partial_func, population_size = 2, candidate_num = 2,
#     sigma=0.1, learning_rate=0.001, reward_goal=40, consecutive_goal_stopping=20,
#     threadcount = 5, cuda=cuda, render_test=False, , task = task_name ,  select_random_parent = False
# )
start = time.time()


final_agent = ne.run(4000, print_step = 1)
end = time.time() - start

pickle.dump(final_agent.weights, open(os.path.abspath(args.weights_path), 'wb'))

partial_func(final_agent, render=False)

print(f"Reward from final weights: {final_agent.reward}")
print(f"Cost from final weights: {final_agent.cost}")
print(f"Time to completion: {end}")