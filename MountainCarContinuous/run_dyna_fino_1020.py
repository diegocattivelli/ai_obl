"""Agrega planning 10 y 20 al experimento fino, hasta 2.000 episodios, con el
mismo protocolo que 0/1/5 (bloques de 100, eval cada bloque, 3 semillas).
No re-corre 0/1/5. Guarda en checkpoints_dyna_fino.

    poetry run python run_dyna_fino_1020.py
"""

import json
import os
import time

import gymnasium as gym

from configs import uniform_pos, nonuniform_vel, make_actions, make_get_state
from dyna_q_agent import DynaQAgent
from train_eval_loop import run_train_eval, aggregate_seed_histories

PLANNING = [10, 20]
SEEDS = [0, 1, 2]
N_CYCLES = 20                 # 20 x 100 = 2.000 episodios
TRAIN_PER_CYCLE = 100         # mismo bloque que 0/1/5
EVAL_EPISODES = 50
OUTDIR = "checkpoints_dyna_fino"

X_SPACE = uniform_pos(20)
V_SPACE = nonuniform_vel()
ACTIONS = make_actions(3)
GET_STATE = make_get_state(X_SPACE, V_SPACE)
N_POS, N_VEL = len(X_SPACE) + 1, len(V_SPACE) + 1
HYPER = dict(alpha=0.1, gamma=0.999, epsilon=0.9, epsilon_min=0.05, epsilon_decay=1.0)

os.makedirs(os.path.join(OUTDIR, "figs"), exist_ok=True)

for k in PLANNING:
    histories = []
    t0 = time.time()
    for seed in SEEDS:
        env = gym.make("MountainCarContinuous-v0")
        agent = DynaQAgent(N_POS, N_VEL, len(ACTIONS), seed=seed,
                           n_planning_steps=k, **HYPER)
        out = run_train_eval(
            agent, env, env, GET_STATE, ACTIONS,
            n_cycles=N_CYCLES, train_episodes_per_cycle=TRAIN_PER_CYCLE,
            eval_episodes=EVAL_EPISODES, config_name="dynafino_plan%d_seed%d" % (k, seed),
            checkpoint_dir=OUTDIR, eval_seed=0, verbose=False)
        histories.append(out["history"])
        env.close()
        print("  planning=%d  seed=%d  listo" % (k, seed))
    agg = aggregate_seed_histories(histories)
    with open(os.path.join(OUTDIR, "dynafino_plan%d_multiseed.json" % k), "w") as f:
        json.dump({"config_name": "dynafino_plan%d" % k, "history": agg,
                   "seeds": SEEDS, "multiseed": True, "n_planning_steps": k}, f, indent=2)
    print("planning=%d: %.1f min | rec final(2000): %d" % (k, (time.time()-t0)/60, round(agg[-1]["mean"])))
print("Listo. Ahora: poetry run python plot_dyna_fino.py")
