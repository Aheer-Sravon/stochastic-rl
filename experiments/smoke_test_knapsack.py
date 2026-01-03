from stochastic_rl.envs.knapsack import StochasticKnapsackEnv

env = StochasticKnapsackEnv(C=50, T=60, w_max=10)

state = env.reset()
done = False

print("Initial state:", state)

while not done:
    c, t, w, v = state

    # baseline policy: accept if it fits
    action = 1 if w <= c else 0

    state, reward, done, info = env.step(action)

    print(
        f"t={t}, w={w}, v={v:.2f}, "
        f"a={action}, accepted={info['accepted']}, "
        f"reward={reward:.2f}, c_left={info['remaining_capacity']:.2f}"
    )

print("\nTotal value collected:", info["total_value"])

