import numpy as np

def simulate_gbm_paths(S0,r,sigma,T,n_steps,n_paths,seed=None):

    if seed is not None: 
        np.random.seed(seed)

    dt=T/n_steps
    paths= np.zeros((n_paths,n_steps+1))
    paths[:,0]=S0

    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + vol * Z)

    return paths

