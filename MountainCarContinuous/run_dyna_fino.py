"""Experimento Dyna-Q FINO: eficiencia de muestras (sample efficiency).

Dyna-Q no busca terminar mas alto sino APRENDER CON MENOS INTERACCIONES REALES,
porque su caso de uso es cuando cada paso en el entorno es caro. Por eso aca
entrenamos pocos episodios (2.000) y evaluamos MUY seguido (cada 50), para ver
la rampa temprana y cuantos episodios reales necesita cada nivel de planning
para llegar a buen rendimiento.

Guarda en una carpeta aparte para no pisar las corridas de 10.000.

    poetry run python run_dyna_fino.py
"""

import json
import os
import time

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting
from configs import uniform_pos, nonuniform_vel, make_actions, make_get_state
from dyna_q_agent import DynaQAgent
from train_eval_loop import run_train_eval, aggregate_seed_histories

# ---- parametros (resolucion fina, pocos episodios) ----
PLANNING = [0, 1, 5, 10, 20]
SEEDS = [0, 1, 2]
N_CYCLES = 40                 # 40 x 50 = 2.000 episodios
TRAIN_PER_CYCLE = 50          # bloque chico -> checkpoint cada 50 episodios
EVAL_EPISODES = 50
OUTDIR = "checkpoints_dyna_fino"

# ---- discretizacion e hiperparametros BASE ----
X_SPACE = uniform_pos(20)
V_SPACE = nonuniform_vel()
ACTIONS = make_actions(3)
GET_STATE = make_get_state(X_SPACE, V_SPACE)
N_POS, N_VEL = len(X_SPACE) + 1, len(V_SPACE) + 1
HYPER = dict(alpha=0.1, gamma=0.999, epsilon=0.9, epsilon_min=0.05, epsilon_decay=1.0)

os.makedirs(os.path.join(OUTDIR, "figs"), exist_ok=True)
agregados, tiempos = [], {}

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
    tiempos[k] = time.time() - t0
    agg = aggregate_seed_histories(histories)
    label = "planning = %d%s" % (k, " (= Q-Learning)" if k == 0 else "")
    agregados.append({"config_name": label, "history": agg})
    with open(os.path.join(OUTDIR, "dynafino_plan%d_multiseed.json" % k), "w") as f:
        json.dump({"config_name": "dynafino_plan%d" % k, "history": agg,
                   "seeds": SEEDS, "multiseed": True, "n_planning_steps": k}, f, indent=2)
    ep90 = next((h["trained_episodes"] for h in agg if h["mean"] >= 90), None)
    print("planning=%d: %.1f min | primer checkpoint con rec>=90: %s" % (k, tiempos[k]/60, ep90))

# ---- figura: rampa de aprendizaje (recompensa vs episodios reales) ----
fig, ax = plt.subplots(figsize=(12, 6))
plotting.plot_comparison_multiseed(
    agregados, ax=ax,
    title="Dyna-Q - eficiencia de muestras (evaluacion fina cada 50 episodios)")
ax.axhline(90, color="gray", ls="--", lw=1, alpha=0.6)
MAXEP = N_CYCLES * TRAIN_PER_CYCLE
ax.set_xticks(range(0, MAXEP + 1, 200))
ax.set_xlim(0, MAXEP)
fig.savefig(os.path.join(OUTDIR, "figs", "dyna_fino.png"), dpi=120, bbox_inches="tight")
print("Guardado:", os.path.join(OUTDIR, "figs", "dyna_fino.png"))
