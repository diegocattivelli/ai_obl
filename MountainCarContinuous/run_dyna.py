"""Experimento Dyna-Q: efecto de los pasos de planificacion.

Corre Dyna-Q sobre la discretizacion BASE (uniforme 20 pos, no uniforme vel, 3
acciones, alfa 0.1, gamma 0.999, epsilon 0.9 fijo) variando UN factor: la
cantidad de pasos de planificacion. Con planning=0 es exactamente Q-Learning.

Para cada nivel de planning entrena 3 semillas con el loop train/test (mismas
curvas de aprendizaje que el resto del trabajo), agrega la banda entre semillas,
guarda el _multiseed.json y ademas registra el TIEMPO de ejecucion de cada nivel
(la consigna pide reportar tiempo). Al final arma dos figuras: la comparativa de
recompensa y la de tiempo vs planning.

    poetry run python run_dyna.py
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

# ---- parametros del experimento (editables) ----
PLANNING = [0, 1, 5, 10, 20, 50]
SEEDS = [0, 1, 2]
N_CYCLES = 10                 # 10 x 1000 = 10.000 episodios
TRAIN_PER_CYCLE = 1000
EVAL_EPISODES = 100
OUTDIR = "checkpoints_dyna"

# ---- discretizacion e hiperparametros BASE ----
X_SPACE = uniform_pos(20)
V_SPACE = nonuniform_vel()
ACTIONS = make_actions(3)
GET_STATE = make_get_state(X_SPACE, V_SPACE)
N_POS, N_VEL = len(X_SPACE) + 1, len(V_SPACE) + 1
HYPER = dict(alpha=0.1, gamma=0.999, epsilon=0.9, epsilon_min=0.05, epsilon_decay=1.0)

os.makedirs(os.path.join(OUTDIR, "figs"), exist_ok=True)

agregados = []     # {config_name, history} por nivel de planning (para la comparativa)
tiempos = {}       # planning -> segundos totales (suma de las 3 semillas)

for k in PLANNING:
    histories = []
    t0 = time.time()
    for seed in SEEDS:
        env = gym.make("MountainCarContinuous-v0")
        agent = DynaQAgent(N_POS, N_VEL, len(ACTIONS), seed=seed,
                           n_planning_steps=k, **HYPER)
        name = f"dyna_plan{k}_seed{seed}"
        out = run_train_eval(
            agent, env, env, GET_STATE, ACTIONS,
            n_cycles=N_CYCLES, train_episodes_per_cycle=TRAIN_PER_CYCLE,
            eval_episodes=EVAL_EPISODES, config_name=name,
            checkpoint_dir=OUTDIR, eval_seed=0, verbose=False,
        )
        histories.append(out["history"])
        env.close()
        print(f"  planning={k:>3}  seed={seed}  listo")
    elapsed = time.time() - t0
    tiempos[k] = elapsed

    agg = aggregate_seed_histories(histories)
    label = f"planning = {k}" + (" (= Q-Learning)" if k == 0 else "")
    agregados.append({"config_name": label, "history": agg})

    with open(os.path.join(OUTDIR, f"dyna_plan{k}_multiseed.json"), "w") as f:
        json.dump({"config_name": f"dyna_plan{k}", "history": agg,
                   "seeds": SEEDS, "multiseed": True, "n_planning_steps": k}, f, indent=2)
    print(f"planning={k:>3}  ->  {elapsed/60:.1f} min  "
          f"(rec final ~{agg[-1]['mean']:.0f}, exito {agg[-1]['success_rate']*100:.0f}%)")

# ---- figura 1: comparativa de recompensa ----
fig, ax = plt.subplots(figsize=(11, 6))
plotting.plot_comparison_multiseed(
    agregados, ax=ax,
    title="Dyna-Q — efecto de los pasos de planificacion (banda entre semillas)")
fig.savefig(os.path.join(OUTDIR, "figs", "dyna_comparacion.png"), dpi=120, bbox_inches="tight")
print("Guardado:", os.path.join(OUTDIR, "figs", "dyna_comparacion.png"))

# ---- figura 2: tiempo de ejecucion vs planning ----
ks = sorted(tiempos)
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(ks, [tiempos[k] / 60 for k in ks], marker="o")
ax2.set_xlabel("Pasos de planificacion")
ax2.set_ylabel("Tiempo de entrenamiento (min, 3 semillas)")
ax2.set_title("Dyna-Q — costo de computo vs pasos de planificacion")
ax2.grid(True, alpha=0.3)
fig2.savefig(os.path.join(OUTDIR, "figs", "dyna_tiempos.png"), dpi=120, bbox_inches="tight")
print("Guardado:", os.path.join(OUTDIR, "figs", "dyna_tiempos.png"))

# ---- resumen de tiempos a CSV ----
with open(os.path.join(OUTDIR, "dyna_tiempos.csv"), "w") as f:
    f.write("planning,segundos,minutos\n")
    for k in ks:
        f.write(f"{k},{tiempos[k]:.1f},{tiempos[k]/60:.2f}\n")
print("Tiempos:", {k: round(tiempos[k]/60, 1) for k in ks}, "min")
