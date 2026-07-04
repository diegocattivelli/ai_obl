"""Dyna-Q sobre la configuracion FINAL (posicion no uniforme izquierda + epsilon
con decaimiento rapido), para ver si con un modelo/exploracion mejores la
planificacion cumple el beneficio esperado, a diferencia de la base.

Mismo protocolo fino: bloques de 100, eval cada bloque, 3 semillas, 2.000 episodios.
Planning 0/1/5. Guarda en checkpoints_dyna_final.

    poetry run python run_dyna_final.py
"""
import json, os, time
import gymnasium as gym

from configs import nonuniform_pos_left, nonuniform_vel, make_actions, make_get_state
from dyna_q_agent import DynaQAgent
from train_eval_loop import run_train_eval, aggregate_seed_histories

PLANNING = [0, 1, 5]
SEEDS = [0, 1, 2]
N_CYCLES = 20                 # 20 x 100 = 2.000 episodios
TRAIN_PER_CYCLE = 100
EVAL_EPISODES = 50
OUTDIR = "checkpoints_dyna_final"

# ---- config FINAL: posicion no uniforme izquierda, velocidad no uniforme, 3 acc ----
X_SPACE = nonuniform_pos_left()
V_SPACE = nonuniform_vel()
ACTIONS = make_actions(3)
GET_STATE = make_get_state(X_SPACE, V_SPACE)
N_POS, N_VEL = len(X_SPACE) + 1, len(V_SPACE) + 1
# epsilon con decaimiento rapido (1.0 -> 0.05, factor 0.9995), igual que la final
HYPER = dict(alpha=0.1, gamma=0.999, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995)

os.makedirs(os.path.join(OUTDIR, "figs"), exist_ok=True)
agregados = []

for k in PLANNING:
    histories = []
    t0 = time.time()
    for seed in SEEDS:
        env = gym.make("MountainCarContinuous-v0")
        agent = DynaQAgent(N_POS, N_VEL, len(ACTIONS), seed=seed,
                           n_planning_steps=k, **HYPER)
        out = run_train_eval(agent, env, env, GET_STATE, ACTIONS,
            n_cycles=N_CYCLES, train_episodes_per_cycle=TRAIN_PER_CYCLE,
            eval_episodes=EVAL_EPISODES, config_name="dynafinal_plan%d_seed%d" % (k, seed),
            checkpoint_dir=OUTDIR, eval_seed=0, verbose=False)
        histories.append(out["history"])
        env.close()
        print("  planning=%d  seed=%d  listo" % (k, seed))
    agg = aggregate_seed_histories(histories)
    agregados.append((k, agg))
    with open(os.path.join(OUTDIR, "dynafinal_plan%d_multiseed.json" % k), "w") as f:
        json.dump({"config_name": "dynafinal_plan%d" % k, "history": agg,
                   "seeds": SEEDS, "multiseed": True, "n_planning_steps": k}, f, indent=2)
    ep90 = next((h["trained_episodes"] for h in agg if h["mean"] >= 90), None)
    print("planning=%d: %.1f min | primer>=90: %s | final: %d" % (k, (time.time()-t0)/60, ep90, round(agg[-1]["mean"])))

# ---- figura: solo medias, sin bandas (legibilidad) ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 6))
for k, agg in agregados:
    x = [h["trained_episodes"] for h in agg]; y = [h["mean"] for h in agg]
    lab = "planning = %d%s" % (k, " (= Q-Learning)" if k == 0 else "")
    ax.plot(x, y, marker="o", ms=4, lw=1.8, label=lab)
ax.axhline(90, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_xlim(0, 2000); ax.set_xticks(range(0, 2001, 200))
ax.set_xlabel("Episodios de entrenamiento")
ax.set_ylabel("Recompensa de test (media entre semillas)")
ax.set_title("Dyna-Q sobre la config FINAL (posicion izq + epsilon decay), sin bandas")
ax.grid(True, alpha=0.3); ax.legend()
out = os.path.join(OUTDIR, "figs", "dyna_final_2000.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("Guardado:", out)
