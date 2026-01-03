def greedy_accept_if_fits(state):
    """
    Accept the item if it fits in remaining capacity.
    """

    c, _, w, _ = state
    return 1 if w <= c else 0
