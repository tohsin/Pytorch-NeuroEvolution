
import copy
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import pickle
import time
from torch import FloatTensor
import random
import numpy as np
import torch
from operator import add
import sys
import wandb

class SafeAgent:
    def __init__(self) -> None:
        self.cost = 0
        self.reward = 0
        self.avg_reward = 0
        self.avg_cost = 0
        self.weights = []

    def set_performance(self, reward, cost = 0):
        self.reward = reward
        self.cost = cost

    def reset_avg(self):
        self.avg_reward = 0
        self.avg_cost = 0

    def add_to_avg(self):
        self.avg_reward += self.reward
        self.avg_cost += self.cost

    def compute_avg(self, n):
        self.avg_reward  /= n
        self.avg_cost  /= n
    

class SafeNeuroEvolution:

    def __init__(
        self, 
        weights, 
        performance_func,
        population_size=50,
        sigma=0.01,
        learning_rate=0.001,
        decay=1.0,
        sigma_decay=1.0,
        threadcount=4,
        render_test=False,
        cuda=False,
        reward_goal=None,
        consecutive_goal_stopping=None,
        save_path=None,
        candidate_num = 10,
        cand_test_time = 10,
        method = 2,
        seeded_env=-1,
        task = ""
    ):
        np.random.seed(int(time.time()))
        self.cand_test_times = cand_test_time
        self.weights = weights
        self.performance_func = performance_func
        self.candidate_num = max(1, candidate_num)
        self.POPULATION_SIZE = max(population_size, self.candidate_num)
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.decay = decay
        self.sigma_decay = sigma_decay
        if cuda and torch.cuda.is_available():
            self.pool = ThreadPool(threadcount)
        else:
            self.pool = Pool(threadcount)
        self.render_test = render_test
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0
        self.save_path = save_path
        self.method = method
        self.seeded_env=seeded_env
        self.task = task

    # def reward_func_wrapper(self):
    def compute_avg(self, agent):
        agent.compute_avg(self.cand_test_times)

    def mutate(self, parent_list, sigma):
        '''
        Creates a new child from single parent  weights
        '''
        child = SafeAgent()
        # loop through weights
        for weight in parent_list[0].weights:
            child.weights.append(weight + sigma *  torch.from_numpy(np.random.normal(0, 0.2, weight.shape)).type(torch.FloatTensor).to(self.device))
            # child = (torch.from_numpy(np.random.randint(0,2,parent.shape))).type(torch.DoubleTensor).to(self.device)
        return child

    def _get_config(self, print_step):
        return{
            "print_step" : print_step,
            "population_size" : self.POPULATION_SIZE,
            "sigma" : self.SIGMA,
            "task" : self.task,
            "candidate_num" : self.candidate_num,
            "method" : self.method,
            "use_cuda" : torch.cuda.is_available()
         }
    
    def run(self, iterations, print_step=10):
        if sys.platform == 'linux': #not debugging on mac but running experiment
            run = wandb.init(project='Safe RL via NeuroEvolution', config = self._get_config(print_step) )

        for iteration in range(iterations):
            n_pop = []
            # create a new agent or mutate old agents
            for i in range(self.POPULATION_SIZE):
                if iteration == 0:
                    safe_agent = SafeAgent()
                    for param in self.weights:
                        safe_agent.weights.append(torch.from_numpy(np.random.randn(*param.data.size())).type(torch.FloatTensor).to(self.device))
                    # n_pop.append([x, 0, i]) # weights, score,  index 
                    n_pop.append([safe_agent, i]) # here we store the agent object alone as it will hold informationa about the reward and cost
                else:
                    # p_id = random.randint(0, self.POPULATION_SIZE-1)
                    p_id = i
                    new_p = self.mutate(pop[p_id], self.SIGMA)
                    n_pop.append([copy.deepcopy(new_p), i]) # copy the object just incase python has attachment to the adress
            
          
            self.pool.map(
                self.performance_func,  [p[0] for p in n_pop] #p[0] is the agent
            )
            
            n_pop.sort(key=lambda p: p[0].reward, reverse=True)

            for i in range(self.candidate_num):
                n_pop[i][1] = i

            if self.seeded_env >= 0:
                if iteration==0:
                    elite=n_pop[0]
                else:
                    elite = max([n_pop[0], prev_elite], key=lambda p: p[1])
            else:
                if iteration == 0:
                    elite_c = n_pop[:self.candidate_num]
                else:
                    elite_c = n_pop[:self.candidate_num-1] + [prev_elite]


                # for eval the elite population 

                # reset avg score of each agent first
                # try to use pool to optimise later
                for agent_ in elite_c:
                    agent : SafeAgent = agent_[0]
                    agent.reset_avg()

                for _ in range(self.cand_test_times):
                    # first compute the rewards which is stored in the agents reward and cost property
                    self.pool.map(
                        self.performance_func, [p[0] for p in elite_c]
                    )

                    # now we add that reward to their avg property
                    # try to use pool to optimise later
                    for agent_ in elite_c:
                        #agent : SafeAgent = agent_[0]
                        agent_[0].add_to_avg()

                # now finally we compute avg
                for agent_ in elite_c:
                        #agent : SafeAgent = agent_[0]
                        agent.compute_avg(self.cand_test_times)

                elite = max(elite_c, key=lambda p: p[0].avg_reward)

            if self.method==1:
                n_pop[elite[2]] = elite
            else:
                if iteration != 0:
                    n_pop[-1] = prev_elite
            pop = n_pop
            prev_elite = elite
            prev_elite[1] = -1 # change index here from 2 to 1


            self.performance_func(
                elite[0], render=self.render_test
            )
            test_reward = elite[0].reward
            test_cost = elite[0].cost

            if (iteration+1) % print_step == 0:
                scalers = {
                    'test_reward': test_reward,
                    "test_cost" : test_cost
                }
                if sys.platform == 'linux':
                    wandb.log(scalers, step = iteration )

                print('iter %d. reward: %f' % (iteration+1, test_reward))
                print('iter %d. cost: %f' % (iteration+1, test_cost))
                if self.save_path:
                    pickle.dump(self.weights, open(self.save_path, 'wb'))
                
            if self.reward_goal and self.consecutive_goal_stopping:
                if test_reward >= self.reward_goal:
                    self.consecutive_goal_count += 1
                else:
                    self.consecutive_goal_count = 0

                if self.consecutive_goal_count >= self.consecutive_goal_stopping:
                    return elite[0]

        return elite[0]