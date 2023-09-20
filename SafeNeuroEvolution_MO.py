
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




class NeuroEvolution:

    def __init__(
        self, 
        weights, 
        reward_func,
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
        task = '',
        select_random_parent = False,
        safety_budget = 25
    ):
        np.random.seed(int(time.time()))
        self.cand_test_times = cand_test_time
        self.weights = weights
        self.reward_function = reward_func
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
        self.select_random_parent = select_random_parent
        self.safety_budget = safety_budget

    # def reward_func_wrapper(self):

    def mutate(self, parent_list, sigma):
        child_list = []
        for parent in parent_list[0]:
            child = parent + sigma *  torch.from_numpy(np.random.normal(0,0.2,parent.shape)).type(torch.FloatTensor).to(self.device)
            # child = (torch.from_numpy(np.random.randint(0,2,parent.shape))).type(torch.DoubleTensor).to(self.device)
            child_list.append(child)
        return child_list

    def _get_config(self, print_step):
        return{
            "print_step" : print_step,
            "population_size" : self.POPULATION_SIZE,
            "sigma" : self.SIGMA,
            "task" : self.task,
            "candidate_num" : self.candidate_num,
            "method" : self.method,
            "use_cuda" : torch.cuda.is_available(),
            "Population_select_random_parent": self.select_random_parent,
            "seed_env" : self.seeded_env,
            'file' : "Candidates_safe_agent",
            "safety_budget" : self.safety_budget
         }
    
    def run(self, iterations, print_step=10):
        weight_idx = 0
        reward_idx = 1
        pos_idx = 2
        cost_idx = 3
        fitness_idx = 4

        if sys.platform == 'linux': #not debugging on mac but running experiment
            run = wandb.init(project='Safe RL via NeuroEvolution', config = self._get_config(print_step) )

        for iteration in range(iterations):
            n_pop = []
            # create a new agent or mutate old agents
            for i in range(self.POPULATION_SIZE):
                if iteration == 0:
                    x = []
                    for param in self.weights:
                        x.append(torch.from_numpy(np.random.randn(*param.data.size())).type(torch.FloatTensor).to(self.device))
                        # x.append((torch.from_numpy(np.random.randint(0, 2,param.data.size()))).to(self.device))
                    n_pop.append([x, 0, i, 0, 0]) # 2 extra zeroes here are cost and a new fitness score
                else:
                    # p_id = random.randint(0, self.POPULATION_SIZE-1)
                    p_id = random.randint(0, self.POPULATION_SIZE-1) if self.select_random_parent else i
                    new_p = self.mutate(pop[p_id], self.SIGMA)
                    n_pop.append([copy.deepcopy(new_p), 0, i, 0, 0])  #2 extra zeroes here are cost and a new fitness score
            
            rewards = self.pool.map( # rewards unwraps into reward, cost and fitness score
                self.reward_function,
                [p[weight_idx] for p in n_pop]
            )
            
            for i, _ in enumerate(n_pop):
                n_pop[i][reward_idx] = rewards[i][0] # first u
                n_pop[i][cost_idx] = rewards[i][1]  # Assign cost
                n_pop[i][fitness_idx] = rewards[i][2]  # Assign fitness score

            # new code starts here
            '''
            1. first sort by cost to find candidates
            2. find number of candidates that fall below treshold
            '''
            n_pop.sort(key=lambda p: p[cost_idx], reverse=False)

            number_safe_candidates = 0

            for i in range(self.candidate_num):
                if n_pop[i][cost_idx] - self.safety_budget  <= 0:
                    number_safe_candidates += 1
                else:
                    break

            
            # n_pop.sort(key=lambda p: p[1], reverse=True)
            

            # for i in range(self.candidate_num):
            #     n_pop[i][2] = i

            if self.seeded_env >= 0:
                if iteration==0:
                    elite=n_pop[0]
                else:
                    elite = max([n_pop[0], prev_elite], key=lambda p: p[1])
            else:
                if iteration==0:
                    elite_c = n_pop[:self.candidate_num]
                else:
                    #in this case we have more than enough safe agents we reselect our candidates from the safe candidates
                    if number_safe_candidates >= self.candidate_num:
                        
                        elite_safe = n_pop[:number_safe_candidates]
                        elite_safe.sort(key=lambda p: p[reward_idx], reverse=True)

                        elite_c = elite_safe[:self.candidate_num-1] + [prev_elite]
                    # in this case we have some safe agents that are more than 3 but not alot we simply use all for candidate agents
                    # and not exlude the last agent as we dont have enough
                    elif number_safe_candidates < self.candidate_num and number_safe_candidates > 3:

                        elite_c = n_pop[:number_safe_candidates] + [prev_elite]
                    # in this case we dont have any safe agents and we simply selcet the lowest agents
                    else:
                         elite_c = n_pop[:self.candidate_num-1] + [prev_elite]

                   
                rewards_list = np.zeros((10,))
                cost_list = np.zeros((10,))
                fitness_list = np.zeros((10,))

                for _ in range(self.cand_test_times):
                    results = self.pool.map(
                        self.reward_function,
                        [p[weight_idx] for p in elite_c]
                    )

                    rewards, costs, fitness_scores = zip(*results)

                    rewards_list += np.array(rewards)
                    cost_list += np.array(costs)
                    fitness_list += np.array(fitness_scores)


                rewards_list/=self.cand_test_times
                cost_list /= self.cand_test_times
                fitness_list /= self.cand_test_times

                for i, _ in enumerate(elite_c):
                    elite_c[i][reward_idx] = rewards_list[i]
                    elite_c[i][cost_idx] = cost_list[i]  # Update cost
                    elite_c[i][fitness_idx] = fitness_list[i]  # Update fitness score

                # swap between fitness and reward
                elite = max(elite_c, key=lambda p: p[fitness_idx])

            if self.method==1:
                n_pop[elite[2]] = elite
            else:
                if iteration != 0:
                    n_pop[-1] = prev_elite
            pop = n_pop
            prev_elite = elite
            prev_elite[2] = -1


            test_results = self.reward_function(
                elite[0], render=self.render_test
            )
            test_reward = test_results[0]
            test_cost = test_results[1]
            test_fitness = test_results[2]

            if (iteration+1) % print_step == 0:
                scalers = {'test_reward': test_reward, 'test_cost': test_cost, 'test_fitness_score': test_fitness}
                if sys.platform == 'linux':
                    wandb.log(scalers, step = iteration )

                print('iter %d. reward: %f' % (iteration+1, test_reward))
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