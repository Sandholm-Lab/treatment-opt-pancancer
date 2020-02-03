"""
A class which takes as input a list of treatments and parallelizes their evaluation.
"""

import time
from multiprocessing import Manager
from src.env.simulator_env import SimulatorEnv
from src.env.drugs import DRUGS
from src.util.pool_hack import MyPool
from src.util.store import initialize_result_dictionary, update_result_dictionary
import numpy as np

# -------------------------------------------------------------------
# Worker administration
# -------------------------------------------------------------------

def initialize(init_queue):
    global env_id
    global environment
    init = init_queue.get()
    env_id = init[0]
    conf = init[1]
    environment = SimulatorEnv(conf["n_steps"], conf["cell_lines"], conf["max_dosage"], conf["objective"], conf["domain"], conf["scale"])

def eval(treatment_vector):
    global env_id
    global environment

    environment.reset()
    done = False
    i = 0
    while not done:
        obs, reward, done, _ = environment.step(treatment_vector[i * len(DRUGS):(i + 1) * len(DRUGS)])
        i += 1
    return (obs, reward)

def terminate(_):
    global env_id
    global environment
    environment.terminate()
    # give every worker time to terminate
    time.sleep(1.0)
    return env_id

# -------------------------------------------------------------------
# Setup conditions for execution
# -------------------------------------------------------------------

class Evaluator():

    def __init__(self, config, n_envs=2, store=True, repeated=False, allow_pd=True):
        self.config = config
        self.n_envs = n_envs
        self.store = store
        self.repeated = repeated
        self.res_buffers = {}
        self.allow_pd = allow_pd
        for line in self.config["cell_lines"]:
            self.res_buffers[line] = initialize_result_dictionary()
        self.worker_pool = self.initialize_workers(n_envs, config)

    def initialize_workers(self, n_envs, config):
        manager = Manager()
        init_queue = manager.Queue()
        for i in range(n_envs):
            init_queue.put((i, config))
        worker_pool = MyPool(n_envs, initialize, (init_queue,))
        return worker_pool

    def evaluate(self, treatments):
        xs = [t.flatten() for t in treatments]
        if self.repeated:
            for x in xs:
                assert len(x) <= len(DRUGS)
            xs = [np.concatenate([x] * self.config["n_steps"]) for x in xs]

        if not self.allow_pd: # insert zeroi value into first entry of every treatment tupel
            for i, x in enumerate(xs):
                assert len(x) == (len(DRUGS) - 1) * self.config["n_steps"], "Detected dimension mismatch in treatment vector."
                new_x = []
                for j in range(self.config["n_steps"]):
                    new_x += [float(0.0)]
                    new_x += [float(val) for val in x[(j * 6):((j + 1) * 6)]]
                xs[i] = np.array(new_x)

        for x in xs:
            assert len(x) == len(DRUGS) * self.config["n_steps"], "Detected dimension mismatch in treatment vector."
        res = self.worker_pool.map(eval, xs)
        ys = [r[1] for r in res]
        prolifs = [r[0] for r in res]

        if self.store: # We simply buffer all experimental results for a later readout
            for i, line in enumerate(self.config["cell_lines"]):
                rel_prolifs = [p[i] for p in prolifs]
                update_result_dictionary(self.res_buffers[line], xs, rel_prolifs, self.config["max_dosage"], self.config["scale"])

        return ys, prolifs

    def terminate(self):
        ids = self.worker_pool.map(terminate, [None for i in range(self.n_envs)])
        ids.sort()
        assert ids == list(range(self.n_envs)), "Not all environment processes have terminated."
        self.worker_pool.close()
        self.worker_pool.join()

    def get_res_dict(self):
        """Return dictionary with results."""
        assert self.store, "This evaluator was not configured to store experimental logs."
        return self.res_buffers
