from abc import ABC, abstractmethod
import numpy as np
from numpy.random import multivariate_normal
from src.sampler.sampler import TruncatedNormalSampler

EPS = 0
SEED = 23
BURN_IN = 5000

class Domain(ABC):
    @abstractmethod
    def contains(self, x):
        raise NotImplementedError()

    @abstractmethod
    def uniform(self, rel_proliferations, action_dict):
        raise NotImplementedError()

    @abstractmethod
    def normal(self, rel_proliferations, action_dict):
        raise NotImplementedError()

    @abstractmethod
    def center(self):
        raise NotImplementedError()

class UnitSimplex(Domain):
    def __init__(self, dim, seed=23):
        np.random.seed(seed)
        self.dim = dim

    def contains(self, x):
        assert len(x) == self.dim
        return (np.all(x >= -EPS)) and (sum(x) <= 1 + EPS)

    def uniform(self):
        """Samples a point uniformly at random from the unit simplex using the Kraemer Algorithm
        The algorithm is described here: https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf

        Returns:
            sample: A point uniformly sampled from the unit simplex.
        """
        uni = np.random.uniform(size=(self.dim + 1))
        uni = np.sort(uni)
        sample =  np.diff(uni, prepend=0) / uni[-1]
        return sample[:-1]

    def normal(self, mu, sigma):
        # TODO: Use MCMC version of this
        s = multivariate_normal(mu.flatten(), sigma)
        while not self.contains(s):
            s = multivariate_normal(mu.flatten(), sigma)
        return s

    def center(self):
        return np.vstack(np.ones(self.dim) / self.dim) 


class SequentialSimplex(Domain):
    def __init__(self, dim, n_steps, seed=23):
        self.seed = seed
        np.random.seed(seed)
        self.dim = dim * n_steps
        self.n_steps = n_steps
        self.single = UnitSimplex(dim, seed=seed)

    def contains(self, x):
        assert len(x) == self.dim
        for i in range(self.n_steps):
            if not self.single.contains(x[i * self.single.dim:(i + 1) * self.single.dim]):
                return False
        return True

    def uniform(self):
        """Samples a point uniformly at random from the sequential simplex.

        Returns:
            sample: A point uniformly sampled from the sequential simplex.
        """
        sample = np.concatenate([self.single.uniform() for i in range(self.n_steps)])
        assert len(sample) == self.dim, "Sampling process has a mistake."
        return sample

    def normal(self, mu, sigma):
        sampler = TruncatedNormalSampler(mu, sigma, self.seed)
        # constraint all positive
        for i in range(self.dim):
            c = np.zeros(self.dim)
            c[i] = 1
            sampler.add_linear_constraint(c, 0)
        # constraint sum less equal one
        for i in range(self.n_steps):
            c = np.zeros(self.dim)
            c[i * self.single.dim: (i + 1) * self.single.dim] = np.ones(self.single.dim)
            sampler.add_linear_constraint(-c, -1)

        s = sampler.sample_with_burn_in(BURN_IN)
        while not self.contains(s):
            print("Resample")
            s = sampler.sample_with_burn_in(BURN_IN)
        s = np.clip(s, a_min=0, a_max=None)
        self.seed += 1
        return s

    def center(self):
        center = np.concatenate([self.single.center().flatten() for i in range(self.n_steps)])
        return np.vstack(center) 


class Cube(Domain):
    def __init__(self, dim, seed=23):
        np.random.seed(seed)
        self.dim = dim        

    def contains(self, x):
        assert len(x) == self.dim
        return (np.all(x >= -EPS)) and (np.all(x <= 1 + EPS))

    def uniform(self):
        return np.random.uniform(size=self.dim)

    def normal(self, mu, sigma):
        s = multivariate_normal(mu.flatten(), sigma)
        while not self.contains(s):
            s = multivariate_normal(mu.flatten(), sigma)
        return s

    def center(self):
        return np.vstack(np.ones(self.dim) / 2)
 

class SequentialCube(Domain):
    def __init__(self, dim, n_steps, seed=23):
        np.random.seed(seed)
        self.dim = dim * n_steps
        self.n_steps = n_steps
        self.single = Cube(dim, seed=seed)

    def contains(self, x):
        assert len(x) == self.dim
        for i in range(self.n_steps):
            if not self.single.contains(x[i * self.single.dim:(i + 1) * self.single.dim]):
                return False
        return True

    def uniform(self):
        """Samples a point uniformly at random from the sequential simplex.

        Returns:
            sample: A point uniformly sampled from the sequential simplex.
        """
        sample = np.concatenate([self.single.uniform() for i in range(self.n_steps)])
        assert len(sample) == self.dim, "Sampling process has a mistake."
        return sample

    def normal(self, mu, sigma):
        s = multivariate_normal(mu.flatten(), sigma)
        while not self.contains(s):
            s = multivariate_normal(mu.flatten(), sigma)
        return s

    def center(self):
        center = np.concatenate([self.single.center().flatten() for i in range(self.n_steps)])
        return np.vstack(center) 


# -------------------------------------------------------------------------------------------------

def retrieve_domain(domain, dim=7, n_steps=1, seed=23):    
    if domain == "simplex":
        if n_steps == 1:
            return UnitSimplex(dim, seed=seed)
        else:
            return SequentialSimplex(dim, n_steps, seed=seed)
    if domain == "cube":
        if n_steps == 1:
            return Cube(dim, seed=seed)
        else:
            return SequentialCube(dim, n_steps, seed=seed)
    else:
        raise ValueError("Specified domain is unknown.")
