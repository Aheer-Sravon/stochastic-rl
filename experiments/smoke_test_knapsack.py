import json
from datetime import datetime

from stochastic_rl.envs.knapsack import StochasticKnapsackEnv
from stochastic_rl.policies import greedy_accept_if_fits, random_accept

env = StochasticKnapsackEnv(C=50, T=60, w_max=10)

state = env.reset()
done = False

trajectory = []
episode_meta = {
    "env": "StochasticKnapsackEnv",
    "params": {"C": 50, "T": 60, "w_max": 10},
    "started_at": datetime.now().isoformat(timespec="seconds"),
}

print("Initial state:", state)

while not done:
    c, t, w, v = state

    # baseline policy: accept if it fits
    action = random_accept(state, 0.3)

    next_state, reward, done, info = env.step(action)

    # store one transition (s, a, r, s', done, info)
    trajectory.append({
        "t": int(t),
        "state": {"c": float(c), "w": float(w), "v": float(v)},
        "action": int(action),
        "reward": float(reward),
        "next_state": {"c": float(next_state[0]), "w": float(next_state[2]), "v": float(next_state[3])},
        "done": bool(done),
        "info": {
            "accepted": bool(info["accepted"]),
            "remaining_capacity": float(info["remaining_capacity"]),
            "total_value": float(info["total_value"]),
        },
    })

    print(
        f"t={t}, w={w}, v={v:.2f}, "
        f"a={action}, accepted={info['accepted']}, "
        f"reward={reward:.2f}, c_left={info['remaining_capacity']:.2f}"
    )

    state = next_state

episode_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
episode_meta["total_value"] = float(info["total_value"])
episode_meta["num_steps"] = len(trajectory)

data = {
    "meta": episode_meta,
    "trajectory": trajectory,
}

out_path = "knapsack_episode.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("\nTotal value collected:", info["total_value"])
print(f"Saved episode data to: {out_path}")

