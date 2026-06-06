"""Corre el set de configuraciones (agrupado por decisión) con el loop train/test.

Uso típico (en tu máquina, con el .venv de Poetry):

    python run_experiments.py                       # 11 configs de grupos + 2 finales
    python run_experiments.py --group acciones      # solo un grupo
    python run_experiments.py --only base_posU20_velNU_3acc
    python run_experiments.py --cycles 5 --train 200 --eval 20            # prueba rápida
    python run_experiments.py --resume checkpoints/<cfg>_ep10000.pkl --cycles 5
    python make_plots.py                            # regenerar comparativas por grupo

MULTI-SEMILLA (v2.1):
    python run_experiments.py --seeds 0,1,2                 # cada config con 3 semillas
    python run_experiments.py --group gamma --seeds 0,1,2   # un grupo, 3 semillas

Con --seeds, cada config se entrena una vez por semilla y se agrega en un
historial con BANDA ENTRE SEMILLAS (`<cfg>_multiseed.json`). make_plots usa ese
historial si existe. Salidas en --outdir (por defecto 'checkpoints'):
    <cfg>_seed<S>_ep<N>.pkl   modelos por semilla (reanudables)
    <cfg>_multiseed.json      historial agregado entre semillas
    figs/<cfg>_multiseed.png  curva con banda entre semillas + tasa de éxito
    figs/grupo_<grupo>.png    comparativa por eje de decisión
"""

import argparse
import json
import os

import gymnasium as gym

from q_learning_agent import QLearningAgent
from configs import build_configs, configs_in_group, make_get_state, GROUPS
from train_eval_loop import run_train_eval, resume_train_eval, aggregate_seed_histories
import plotting
from make_plots import make_group_plots


ENV_ID = "MountainCarContinuous-v0"


def _make_agent(cfg, seed):
    return QLearningAgent(
        n_pos=len(cfg["x_space"]) + 1,
        n_vel=len(cfg["vel_space"]) + 1,
        n_actions=len(cfg["actions"]),
        seed=seed,
        **cfg["hyper"],
    )


def run_one(cfg, args, seed):
    """Una corrida (una semilla). Genera el reporte de 3 paneles por config."""
    train_env = gym.make(ENV_ID, render_mode="rgb_array")
    eval_env = gym.make(ENV_ID, render_mode="rgb_array")
    get_state = make_get_state(cfg["x_space"], cfg["vel_space"])
    agent = _make_agent(cfg, seed)
    result = run_train_eval(
        agent, train_env, eval_env, get_state, cfg["actions"],
        n_cycles=args.cycles, train_episodes_per_cycle=args.train,
        eval_episodes=args.eval, config_name=cfg["name"],
        checkpoint_dir=args.outdir, eval_seed=args.eval_seed,
    )
    plotting.plot_full_report(result, band=args.band,
                              outdir=os.path.join(args.outdir, "figs"), show=False)
    train_env.close()
    eval_env.close()
    return result


def run_one_multiseed(cfg, args, seeds):
    """Varias semillas de una config -> historial agregado entre semillas."""
    get_state = make_get_state(cfg["x_space"], cfg["vel_space"])
    histories = []
    for s in seeds:
        print(f"  -- semilla {s} --")
        train_env = gym.make(ENV_ID, render_mode="rgb_array")
        eval_env = gym.make(ENV_ID, render_mode="rgb_array")
        agent = _make_agent(cfg, s)
        res = run_train_eval(
            agent, train_env, eval_env, get_state, cfg["actions"],
            n_cycles=args.cycles, train_episodes_per_cycle=args.train,
            eval_episodes=args.eval, config_name=f"{cfg['name']}_seed{s}",
            checkpoint_dir=args.outdir, save_every=args.cycles,  # solo .pkl final por semilla
            eval_seed=args.eval_seed,
        )
        histories.append(res["history"])
        train_env.close()
        eval_env.close()

    agg = aggregate_seed_histories(histories)
    out = {"config_name": cfg["name"], "history": agg,
           "seeds": list(seeds), "multiseed": True}
    with open(os.path.join(args.outdir, f"{cfg['name']}_multiseed.json"), "w") as f:
        json.dump(out, f, indent=2)
    plotting.plot_multiseed_report(out, outdir=os.path.join(args.outdir, "figs"), show=False)
    return agg


def _select_configs(args):
    configs = build_configs()
    if args.only:
        configs = [c for c in configs if c["name"] == args.only]
        if not configs:
            raise SystemExit(f"No existe la config '{args.only}'")
    elif args.group:
        configs = configs_in_group(args.group, configs)
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=10, help="ciclos train+test")
    p.add_argument("--train", type=int, default=1000, help="episodios de train por ciclo")
    p.add_argument("--eval", type=int, default=100, help="episodios de test por checkpoint")
    p.add_argument("--eval-seed", type=int, default=0, help="semilla base de evaluación")
    p.add_argument("--seed", type=int, default=0, help="semilla de entrenamiento (1 corrida)")
    p.add_argument("--seeds", default=None,
                   help="lista de semillas separadas por coma -> modo multi-semilla (ej. 0,1,2)")
    p.add_argument("--band", default="std", choices=["std", "minmax", "iqr"])
    p.add_argument("--outdir", default="checkpoints")
    p.add_argument("--only", default=None, help="correr solo la config con ese nombre")
    p.add_argument("--group", default=None, choices=list(GROUPS.keys()),
                   help="correr solo las configs de un grupo")
    p.add_argument("--resume", default=None, help="reanudar desde un checkpoint .pkl")
    args = p.parse_args()

    if args.resume:
        train_env = gym.make(ENV_ID, render_mode="rgb_array")
        eval_env = gym.make(ENV_ID, render_mode="rgb_array")
        _, extra = QLearningAgent.load(args.resume)
        cfg = next(c for c in build_configs() if c["name"] == extra.get("config_name"))
        get_state = make_get_state(cfg["x_space"], cfg["vel_space"])
        result, _ = resume_train_eval(
            args.resume, train_env, eval_env, get_state, cfg["actions"],
            n_cycles=args.cycles, train_episodes_per_cycle=args.train,
            eval_episodes=args.eval, checkpoint_dir=args.outdir, eval_seed=args.eval_seed,
        )
        plotting.plot_full_report(result, band=args.band,
                                  outdir=os.path.join(args.outdir, "figs"), show=False)
        make_group_plots(args.outdir, band=args.band)
        return

    configs = _select_configs(args)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    for i, cfg in enumerate(configs, 1):
        print("\n" + "=" * 70)
        modo = f"semillas={seeds}" if seeds else f"seed={args.seed}"
        print(f"CONFIG {i}/{len(configs)}: {cfg['name']}  | grupos={cfg['groups']} | {modo}")
        print("=" * 70)
        if seeds:
            run_one_multiseed(cfg, args, seeds)
        else:
            run_one(cfg, args, args.seed)

    print("\nGenerando comparativas por grupo...")
    make_group_plots(args.outdir, band=args.band)


if __name__ == "__main__":
    main()
