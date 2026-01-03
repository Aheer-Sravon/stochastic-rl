import random

def random_accept(state, p=0.5):
    """
    Accept with probability p if item fits.
    """

    c, _, w, _ = state

    if w > c:
        return 0;

    return 1 if random.random() < p else 0
