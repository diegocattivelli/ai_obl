"""Genera las gráficas comparativas POR GRUPO de decisión a partir de los
historiales ya guardados.

Para cada config se usa, si existe, el historial **multi-semilla**
(`<config>_multiseed.json`, banda entre semillas); si no, el de una sola corrida
(`<config>_history.json`). No depende de gymnasium, así que regenera figuras sin
reentrenar:

    python make_plots.py                 # todos los grupos con historial disponible
    python make_plots.py --band minmax   # cambiar el tipo de banda
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting
from configs import GROUPS, configs_in_group


def load_history(outdir, name):
    """Devuelve (history, is_multiseed). Prefiere el json multi-semilla."""
    ms = os.path.join(outdir, f"{name}_multiseed.json")
    if os.path.exists(ms):
        with open(ms) as f:
            return json.load(f)["history"], True
    single = os.path.join(outdir, f"{name}_history.json")
    if os.path.exists(single):
        with open(single) as f:
            return json.load(f)["history"], False
    return None, False


def make_group_plots(outdir="checkpoints", band="std"):
    figs_dir = os.path.join(outdir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    generadas = []

    for group_key, title in GROUPS.items():
        items, any_multiseed = [], False
        for cfg in configs_in_group(group_key):
            hist, is_ms = load_history(outdir, cfg["name"])
            if hist:
                label = cfg["name"] + (" (ganadora)" if "base" in cfg["name"] else "")
                items.append((label, hist))
                any_multiseed = any_multiseed or is_ms

        if len(items) < 2:
            print(f"[saltado] {group_key}: faltan historiales ({len(items)} disponibles)")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        plotting.plot_comparison(items, band=band, ax=ax, title=f"Decisión: {title}")
        if any_multiseed:
            ax.set_ylabel(ax.get_ylabel() + " (media entre semillas)")
        out = os.path.join(figs_dir, f"grupo_{group_key}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        generadas.append(out)
        print("Guardado:", out, "(multi-semilla)" if any_multiseed else "")

    return generadas


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="checkpoints")
    p.add_argument("--band", default="std", choices=["std", "minmax", "iqr"])
    args = p.parse_args()
    make_group_plots(args.outdir, args.band)
