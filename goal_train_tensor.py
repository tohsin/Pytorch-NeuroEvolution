import argparse
import copy
from functools import partial
import logging
import os
import pickle
import time
import sys
from SafeNeuroEvolutionTensor import NeuroEvolution

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
    
task_name  =  "SafetyPointGoal1-v0"
env = safety_gymnasium.make(task_name)
obs_size = env.observation_space.shape[0]
actions_space = 2
model = Agent(obs_size, [32, 32], output_size = actions_space)

if cuda:
    print("using Cuda")
    model = model.to('cuda')
def get_reward(weights, model, render=False):
    '''
    returns reward, cost and weighted score, reward - lamda * cost
    '''
    with torch.no_grad():
        cloned_model = copy.deepcopy(model)
        for i, param in enumerate(cloned_model.parameters()):
            try:
                param.data.copy_(weights[i])
            except:
                param.data.copy_(weights[i].data)
        env = safety_gymnasium.make(task_name)
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
    fitness =   total_reward - 2 * np.log( max(0, total_cost - 25 ) )
    return total_reward, total_cost, fitness
    
partial_func = partial(get_reward, model = model)
mother_parameters = list(model.parameters())

ne = NeuroEvolution(
    mother_parameters, partial_func, population_size=50,
    sigma=0.1, learning_rate=0.001, reward_goal=40, consecutive_goal_stopping=20,
    threadcount=50, cuda=cuda, render_test=False, task = task_name, select_random_parent = False
)

# ne = NeuroEvolution(
#     mother_parameters, partial_func, population_size=10,
#     sigma=0.1, learning_rate=0.001, reward_goal=40, consecutive_goal_stopping=20,
#     threadcount=50, cuda=cuda, render_test=False
# )
start = time.time()


final_weights = ne.run(4000, print_step = 1)
end = time.time() - start

pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))

reward = partial_func(final_weights, render=True)

print(f"Reward from final weights: {reward}")
print(f"Time to completion: {end}")