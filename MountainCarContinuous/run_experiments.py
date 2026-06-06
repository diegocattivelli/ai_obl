"""Corre el set de configuraciones (agrupado por decisión) con el loop train/test.

Uso típico (en tu máquina, con el .venv de Poetry):

    python run_experiments.py                       # las 11 configs
    python run_experiments.py --group acciones      # solo las del grupo acciones
    python run_experiments.py --only base_posU20_velNU_3acc
    python run_experiments.py --cycles 5 --train 200 --eval 20   # prueba rápida
    python run_experiments.py --resume checkpoints/base_posU20_velNU_3acc_ep10000.pkl --cycles 5
    python make_plots.py                            # regenerar comparativas por grupo

Genera, dentro de --outdir (por defecto 'checkpoints'):
    <config>_ep<N>.pkl        modelos computados (reanudables)
    <config>_metrics.csv      una fila por checkpoint
    <config>_history.json     historial completo (recompensas crudas incluidas)
    figs/<config>_reporte.png curva de aprendizaje + éxito + train (por config)
    figs/grupo_<grupo>.png    comparativa por eje de decisión

Sugerencia: como cada config tarda varios minutos, podés correr grupo por grupo
en distintas sesiones; make_plots.py arma las comparativas con lo que haya.
"""

import argparse
import os

import gymnasium as gym

from q_learning_agent import QLearningAgent
from configs import build_configs, configs_in_group, make_get_state, GROUPS
from train_eval_loop import run_train_eval, resume_train_eval
import plotting
from make_plots import make_group_plots


ENV_ID = "MountainCarContinuous-v0"


def run_one(cfg, args):
    train_env = gym.make(ENV_ID, render_mode="rgb_array")
    eval_env = gym.make(ENV_ID, render_mode="rgb_array")
    get_state = make_get_state(cfg["x_space"], cfg["vel_space"])

    agent = QLearningAgent(
        n_pos=len(cfg["x_space"]) + 1,
        n_vel=len(cfg["vel_space"]) + 1,
        n_actions=len(cfg["actions"]),
        seed=args.seed,
        **cfg["hyper"],
    )
    result = run_train_eval(
        agent, train_env, eval_env, get_state, cfg["actions"],
        n_cycles=args.cycles,
        train_episodes_per_cycle=args.train,
        eval_episodes=args.eval,
        config_name=cfg["name"],
        checkpoint_dir=args.outdir,
        eval_seed=args.eval_seed,
    )
    plotting.plot_full_report(result, band=args.band,
                              outdir=os.path.join(args.outdir, "figs"), show=False)
    train_env.close()
    eval_env.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=10, help="ciclos train+test")
    p.add_argument("--train", type=int, default=1000, help="episodios de train por ciclo")
    p.add_argument("--eval", type=int, default=100, help="episodios de test por checkpoint")
    p.add_argument("--eval-seed", type=int, default=0, help="semilla base de evaluación")
    p.add_argument("--seed", type=int, default=0, help="semilla de entrenamiento")
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

    configs = build_configs()
    if args.only:
        configs = [c for c in configs if c["name"] == args.only]
        if not configs:
            raise SystemExit(f"No existe la config '{args.only}'")
    elif args.group:
        configs = configs_in_group(args.group, configs)

    for i, cfg in enumerate(configs, 1):
        print("\n" + "=" * 70)
        print(f"CONFIG {i}/{len(configs)}: {cfg['name']}  | grupos={cfg['groups']}")
        print("=" * 70)
        run_one(cfg, args)

    # Comparativas por grupo (usa todos los historiales disponibles en outdir).
    print("\nGenerando comparativas por grupo...")
    make_group_plots(args.outdir, band=args.band)


if __name__ == "__main__":
    main()
