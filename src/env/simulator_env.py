"""
This is the main class for all optimization experiments. It takes a list of cells and parallizes the execution.
It uses a reward and a penalty function to convert an array of proliferation values to a single objective value. 
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import numpy as np
from scipy.stats import lognorm
from pprint import pprint
import gym
from gym import spaces
from reference_simulator.simulator import Simulator
from src.env.drugs import empty_treatment
from util.prepare_dict import prepare_dict
from multiprocessing import Manager
from src.util.pool_hack import MyPool

EPS = 1e-6

# -------------------------------------------------------------------
# Worker administration
# -------------------------------------------------------------------

def initialize(init_queue): # TODO: Make sure it is one simulator per cell line and no switches
    global cell_line
    global simulator
    cell_line = init_queue.get()
    simulator = Simulator()

def reset_worker(line): # TODO: Implement reset of simulator
    global cell_line
    global simulator
    simulator.initialized = False
    obs = simulator.initialize(cell_line)
    return (cell_line, obs) # TODO: Why do we return the cell line here?

def execute_experiment(treatment):
    global cell_line
    global simulator
    rel_proliferation = simulator.apply_treatment(treatment)
    return (cell_line, rel_proliferation)

# -------------------------------------------------------------------
# Setup conditions for experiments
# -------------------------------------------------------------------

class SimulatorEnv(gym.Env):
    """
    This class implements a gym interface to run experiments with the cell simulator. It supports single
    and multi-cell-line experiments. In order to cover various treatment scenarious one can supply custom
    reward functions that take a vector of relative proliferation rates as input. 
     """

    def __init__(self, n_steps, cell_lines, max_dosage, objective, domain, scale):
        """Initializes a new bio-steering environment.

        Args:
            n_steps: Length of sequential treatment plan.
        """

        self.cell_lines = cell_lines
        self.max_dosage = max_dosage
        self.objective = objective
        self.scale = scale

        self.worker_pool = self.initialize_workers(cell_lines)

        self.num_actions = 7 # there are 7 drugs
        self.n_steps = n_steps
        self.step_counter = 0
        self.commulative_treatment = empty_treatment()

        # NOTE: Usually a gym interface would have an action_space attribute
        # here we work with a custom domain class
        if domain.dim > 7: # this extracts the domain for the individual step
            self.domain = domain.single
        else: 
            self.domain = domain 
         

        # NOTE: For now we return the proliferation values as observation
        self.observation_space = spaces.Box( 
            low=np.zeros(1), 
            high= np.ones(1),
            dtype=np.float32
        )

    def initialize_workers(self, cell_lines):
        manager = Manager()
        cellQueue = manager.Queue()
        for cell_line in cell_lines:
            cellQueue.put(cell_line)
        worker_pool = MyPool(len(cell_lines), initialize, (cellQueue,))
        return worker_pool

    def terminate(self):
        # Python guaranetees closure of all processes.
        self.worker_pool.close()
        self.worker_pool.join()

    def sort_by_cell_line(self, results):
        vs = []
        for line in self.cell_lines:
            for entry in results:
                if line == entry[0]:
                    vs.append(entry[1])
        assert len(vs) == len(self.cell_lines)
        return np.array(vs)

    def reset(self):
        '''
        Resets the state of the environment.
        
        :return observation: Initial observation of the environment.
        '''
        results = self.worker_pool.map(reset_worker, self.cell_lines)
        obs = self.sort_by_cell_line(results)
        self.step_counter = 0
        self.commulative_treatment = empty_treatment()
        return obs

    def step(self, action, verbose=False):
        '''
        Run one time step of the environment's dynamics. The actions is assumed to be a flat numpy
        array or a list.
        
        :param action: an action provided by the environment
        :return observation: agent's observation of the current environment
        :return reward: amount of reward returned after previous action
        :return done: whether the episode has ended, in which case further step() calls will return undefined results
        :return info: contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        '''  
        assert self.step_counter < self.n_steps, "Environment has already terminated."
        assert self.domain.contains(action), "The provided actions does not belong to the domain."

        action_dict = prepare_dict(action,  max_dosage=self.max_dosage, scale=self.scale)
        for k in action_dict:
            self.commulative_treatment[k] += action_dict[k]
        jobs = [action_dict for _ in range(len(self.cell_lines))]
        
        results = self.worker_pool.map(execute_experiment, jobs)
        rel_proliferations = self.sort_by_cell_line(results)
        # NOTE: For now we return the proliferation values as observation
        obs = np.array(rel_proliferations)
        reward = self.objective.eval(rel_proliferations, self.commulative_treatment)

        self.step_counter += 1
        if self.step_counter < self.n_steps:
            return obs, reward, False, {}
        else:
            return obs, reward, True, {}
