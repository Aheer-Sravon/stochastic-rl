import numpy as np

def sample_knapsack_item(w_max):
    """
    Sample one item (w, v).

    w ~ UniformInt(1, w_max)
    v = 2*w + noise, noise ~ UniformInt(-2, 8)
    v is clipped to be >= 1
    """

    w = np.random.randint(1, w_max + 1)
    noise = np.random.normal(0.0, 2)
    v = 2*w + noise
    v = max(v, 1)

    return w, v
