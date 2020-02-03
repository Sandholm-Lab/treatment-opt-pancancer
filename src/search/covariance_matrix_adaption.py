"""
Below an implementation of the covariance-matrix adaption method following the book "Algorithms for Optimization" p. 138
"""

import numpy as np



def cma_es(evaluator, domain, max_iter, verbose=True, seed=23):
    np.random.seed(seed)

    n = domain.dim
    sigma = 0.25 # 1.0
    m = int(4 + np.floor(3 * np.log(n))) # recommended value
    m_elite = int(np.floor(m / 2)) # recommended value
    mu = domain.center() # initialize in the center of the domain

    # constants
    ws = [np.log((m + 1) / 2) - np.log(i) for i in range(1, m_elite + 1)] + [0 for _ in range(m - m_elite)]
    ws = np.vstack(ws) / sum(ws) # normalized version

    mu_eff = 1 / (ws.T @ ws)[0,0]
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_S = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + (1 / mu_eff)) / ((n + 2) ** 2 + mu_eff))
    E = np.sqrt(n) * (1 - (1 / (4 * n)) + 1 / (21 * n * n))
    p_sigma = np.vstack(np.zeros(n)) 
    p_S = np.vstack(np.zeros(n))
    S = np.identity(n)
    xs = [None] * m # container for samples

    for k in range(1, max_iter + 1):
        # sample and evaluate
        for i in range(m):
            xs[i] = domain.normal(mu, sigma * sigma * S)
        ys, _ = evaluator.evaluate(xs)
        ids = np.argsort(ys) 

        if verbose:
            # TODO: put some effort to make this look nice
            avg_elite = sum([ys[ids[i]] for i in range(m_elite)]) / m_elite
            print(k, ":", "Average: ", avg_elite, "mu: ", mu.flatten(), sum(mu))

        # selection and mean update
        delta_s = [(np.vstack(x) - mu) / sigma for x in xs]
        delta_w = sum([ws[i] * delta_s[ids[i]] for i in range(m_elite)])
        mu += sigma * delta_w

        # step-size control
        values, vectors = np.linalg.eig(S)
        Q = vectors
        D = np.sqrt(np.diag(values))
        C = Q @ np.linalg.inv(D) @ Q.T
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C @ delta_w
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E - 1)) 

        # covariance adaption
        h_sigma = int((np.linalg.norm(p_sigma) / np.sqrt(1 - ((1 - c_sigma) ** (2 * k)))) < ((1.4 + 2 / (n + 1)) * E))
        p_S = (1 - c_S) * p_S + h_sigma * np.sqrt(c_S * (2 - c_S) * mu_eff) * delta_w

        w0 = [ws[i] if ws[i] >= 0 else (n * ws[i] / (np.linalg.norm(C @ delta_s[ids[i]]) ** 2)) for i in range(m)]

        S = (1 - c_1 - c_mu) * S + c_1 * (p_S @ p_S.T + (1 - h_sigma) * c_S * (2 - c_S) * S) \
            + c_mu * sum(w0[i] * delta_s[ids[i]] @ delta_s[ids[i]].T for i in range(m_elite))
        S = np.triu(S) + np.triu(S, 1).T # enforce symmetry

    obj, prolif = evaluator.evaluate([mu])
    return mu.flatten(), obj[0], prolif[0]
