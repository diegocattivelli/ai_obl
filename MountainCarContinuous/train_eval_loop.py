"""Harness de entrenamiento/evaluación intercalado para Q-Learning.

Idea central (metodología pedida por el profesor):
    en vez de entrenar TODO y testear UNA sola vez al final, se hace un loop:

        train K episodios -> test M episodios -> guardar métricas y checkpoint
        train K episodios -> test M episodios -> ...

    Así se obtiene una CURVA DE APRENDIZAJE: cómo evoluciona el desempeño del
    agente a medida que entrena, con su variabilidad (no un único número final).

Cada checkpoint registra, sobre los M episodios de test:
    media, desvío, min, max, percentiles 25/50/75, tasa de éxito y pasos medios.

Además guarda la tabla Q en .pkl para poder REANUDAR el entrenamiento.
"""

import csv
import json
import os

import numpy as np

from q_learning_agent import QLearningAgent


def _summarize(values):
    """Estadísticos robustos de una lista de recompensas de test."""
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
    }


def run_train_eval(
    agent,
    train_env,
    eval_env,
    get_state,
    actions,
    n_cycles=10,
    train_episodes_per_cycle=1000,
    eval_episodes=100,
    config_name="config",
    checkpoint_dir="checkpoints",
    save_every=1,
    eval_seed=0,
    eval_at_start=True,
    verbose=True,
):
    """Corre el loop train/test y devuelve el historial de checkpoints.

    Parámetros
    ----------
    agent : QLearningAgent
        Agente ya construido (con su discretización e hiperparámetros).
    train_env, eval_env : gym.Env
        Entornos para entrenar y para evaluar (pueden ser el mismo; conviene
        uno aparte en modo 'rgb_array').
    get_state : callable(obs) -> (pos_bin, vel_bin)
    actions : list[float]
        Acciones discretas.
    n_cycles : int
        Cantidad de ciclos train+test.
    train_episodes_per_cycle : int
        Episodios de entrenamiento por ciclo (ej. 1000).
    eval_episodes : int
        Episodios de evaluación por checkpoint (ej. 100).
    eval_seed : int | None
        Semilla base para evaluar siempre los mismos inicios (curvas
        comparables). None => inicios aleatorios.
    eval_at_start : bool
        Si True, registra un checkpoint en 0 episodios (línea base sin
        entrenar) antes del primer ciclo.

    Retorna
    -------
    dict con:
        'config_name', 'history' (lista de dicts por checkpoint),
        'train_rewards' (todas las recompensas de train concatenadas),
        'checkpoint_path' (último .pkl guardado).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []
    train_rewards_all = []
    last_ckpt = None

    def _do_eval(tag):
        ev = agent.evaluate(eval_env, get_state, actions, eval_episodes, seed=eval_seed)
        rec = {
            "trained_episodes": agent.trained_episodes,
            "epsilon": agent.epsilon,
            "test_rewards": ev["rewards"],
            "success_rate": float(np.mean(ev["successes"])),
            "mean_steps": float(np.mean(ev["steps"])),
            **_summarize(ev["rewards"]),
        }
        history.append(rec)
        if verbose:
            print(
                f"[{config_name}] {tag} | episodios={rec['trained_episodes']:>6} "
                f"| test mean={rec['mean']:7.2f} ± {rec['std']:6.2f} "
                f"| [{rec['min']:7.2f}, {rec['max']:7.2f}] "
                f"| éxito={rec['success_rate']*100:5.1f}% "
                f"| pasos={rec['mean_steps']:6.1f} | eps={rec['epsilon']:.3f}"
            )
        return rec

    # Línea base (sin entrenar).
    if eval_at_start:
        _do_eval("baseline")

    for cycle in range(1, n_cycles + 1):
        tr = agent.train(train_env, get_state, actions, train_episodes_per_cycle)
        train_rewards_all.extend(tr["rewards"])
        _do_eval(f"ciclo {cycle}/{n_cycles}")

        if save_every and (cycle % save_every == 0 or cycle == n_cycles):
            last_ckpt = os.path.join(
                checkpoint_dir, f"{config_name}_ep{agent.trained_episodes}.pkl"
            )
            agent.save(
                last_ckpt,
                extra={
                    "config_name": config_name,
                    "history": history,
                    "train_rewards": train_rewards_all,
                    "train_episodes_per_cycle": train_episodes_per_cycle,
                    "eval_episodes": eval_episodes,
                    "eval_seed": eval_seed,
                },
            )

    # CSV resumen (una fila por checkpoint).
    _write_csv(os.path.join(checkpoint_dir, f"{config_name}_metrics.csv"), history)
    # JSON con todo el historial (incluye las recompensas crudas de cada test).
    with open(os.path.join(checkpoint_dir, f"{config_name}_history.json"), "w") as f:
        json.dump({"config_name": config_name, "history": history}, f, indent=2)

    return {
        "config_name": config_name,
        "history": history,
        "train_rewards": train_rewards_all,
        "checkpoint_path": last_ckpt,
    }


def resume_train_eval(
    checkpoint_path,
    train_env,
    eval_env,
    get_state,
    actions,
    n_cycles,
    train_episodes_per_cycle=1000,
    eval_episodes=100,
    checkpoint_dir="checkpoints",
    save_every=1,
    eval_seed=0,
    verbose=True,
):
    """Reanuda el loop desde un checkpoint .pkl y continúa entrenando.

    Reconstruye el agente (tabla Q, epsilon, episodios entrenados) y el
    historial previo, y corre `n_cycles` ciclos adicionales.
    """
    agent, extra = QLearningAgent.load(checkpoint_path)
    config_name = extra.get("config_name", "config")
    if verbose:
        print(
            f"Reanudando '{config_name}' desde {os.path.basename(checkpoint_path)} "
            f"({agent.trained_episodes} episodios ya entrenados)."
        )

    out = run_train_eval(
        agent,
        train_env,
        eval_env,
        get_state,
        actions,
        n_cycles=n_cycles,
        train_episodes_per_cycle=train_episodes_per_cycle,
        eval_episodes=eval_episodes,
        config_name=config_name,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every,
        eval_seed=eval_seed,
        eval_at_start=False,  # ya hay historial; no repetir baseline
        verbose=verbose,
    )
    # Anteponer el historial previo.
    out["history"] = extra.get("history", []) + out["history"]
    out["train_rewards"] = extra.get("train_rewards", []) + out["train_rewards"]
    return out, agent


def _write_csv(path, history):
    if not history:
        return
    cols = [
        "trained_episodes", "epsilon", "mean", "std", "min", "max",
        "p25", "p50", "p75", "success_rate", "mean_steps",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for h in history:
            w.writerow([h.get(c) for c in cols])
