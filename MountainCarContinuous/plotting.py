"""Gráficas para los experimentos de Q-Learning.

La idea es mostrar el desempeño NO como un único número final sino como una
curva a lo largo del entrenamiento, con su variabilidad (bandas de rango):

  - banda 'std'    -> media ± desvío estándar
  - banda 'minmax' -> entre el peor y el mejor episodio de test
  - banda 'iqr'    -> entre percentil 25 y 75 (rango intercuartílico)

Todas las funciones aceptan un `ax` opcional y devuelven el eje, para poder
componer figuras o guardarlas desde el notebook.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def _xy_band(history, band="std"):
    """Extrae x (episodios), y (media) y los límites de la banda elegida."""
    x = np.array([h["trained_episodes"] for h in history], dtype=float)
    mean = np.array([h["mean"] for h in history], dtype=float)
    if band == "std":
        std = np.array([h["std"] for h in history], dtype=float)
        low, high = mean - std, mean + std
    elif band == "minmax":
        low = np.array([h["min"] for h in history], dtype=float)
        high = np.array([h["max"] for h in history], dtype=float)
    elif band == "iqr":
        low = np.array([h["p25"] for h in history], dtype=float)
        high = np.array([h["p75"] for h in history], dtype=float)
    else:
        raise ValueError("band debe ser 'std', 'minmax' o 'iqr'")
    return x, mean, low, high


def plot_learning_curve(
    history, band="std", label=None, ax=None, color=None, show_minmax=True
):
    """Curva de aprendizaje: recompensa media de test vs episodios entrenados,
    con banda de variabilidad. Opcionalmente dibuja también el min/max punteado.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    x, mean, low, high = _xy_band(history, band)
    line, = ax.plot(x, mean, marker="o", label=label, color=color)
    c = line.get_color()
    ax.fill_between(x, low, high, alpha=0.20, color=c)

    if show_minmax and band != "minmax":
        xm, _, mn, mx = _xy_band(history, "minmax")
        ax.plot(xm, mn, ls=":", lw=1, color=c, alpha=0.7)
        ax.plot(xm, mx, ls=":", lw=1, color=c, alpha=0.7)

    band_txt = {"std": "media ± std", "minmax": "min–max", "iqr": "P25–P75"}[band]
    ax.set_xlabel("Episodios de entrenamiento")
    ax.set_ylabel("Recompensa de test")
    ax.set_title(f"Curva de aprendizaje ({band_txt}; punteado = min/max)")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_comparison(results, band="std", ax=None, title="Comparación de configuraciones"):
    """Compara varias configuraciones en un mismo eje.

    `results` es una lista de dicts (la salida de run_train_eval) o una lista de
    tuplas (nombre, history).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        if isinstance(r, dict):
            name, history = r.get("config_name"), r["history"]
        else:
            name, history = r
        plot_learning_curve(history, band=band, label=name, ax=ax, show_minmax=False)

    ax.set_title(title)
    ax.legend()
    return ax


def plot_success_rate(history, label=None, ax=None):
    """Tasa de éxito (fracción de episodios de test que llegan a la meta)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    x = [h["trained_episodes"] for h in history]
    y = [h["success_rate"] * 100 for h in history]
    ax.plot(x, y, marker="s", label=label)
    ax.set_xlabel("Episodios de entrenamiento")
    ax.set_ylabel("Tasa de éxito (%)")
    ax.set_ylim(-2, 102)
    ax.set_title("Tasa de éxito en test")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_train_rewards(train_rewards, window=100, label=None, ax=None):
    """Recompensa de entrenamiento por episodio, suavizada (media móvil)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    r = np.asarray(train_rewards, dtype=float)
    ax.plot(r, alpha=0.25, color="gray")
    if len(r) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(r, kernel, mode="valid")
        ax.plot(np.arange(window - 1, len(r)), smooth, label=label or f"media móvil ({window})")
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel("Recompensa")
    ax.set_title("Recompensa de entrenamiento (suavizada)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_full_report(result, band="std", outdir=None, show=True):
    """Figura resumen de una config: curva de aprendizaje + tasa de éxito +
    recompensa de train. Si `outdir` se especifica, guarda un .png.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_learning_curve(result["history"], band=band, label=result["config_name"], ax=axes[0])
    plot_success_rate(result["history"], label=result["config_name"], ax=axes[1])
    plot_train_rewards(result["train_rewards"], ax=axes[2])
    fig.suptitle(f"Reporte — {result['config_name']}", fontsize=14)
    fig.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{result['config_name']}_reporte.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print("Guardado:", path)
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------------- #
# v2.1 — Gráficas multi-semilla
# ----------------------------------------------------------------------------- #
def plot_learning_curve_multiseed(history, label=None, ax=None, color=None, show_seeds=True):
    """Curva de aprendizaje agregada entre semillas.

    `history` es la salida de aggregate_seed_histories: cada checkpoint trae
    `mean`/`std` ENTRE semillas y los valores por semilla (`seed_means`).
    La banda = media ± desvío entre semillas (robustez ante el azar). Si
    show_seeds, dibuja además cada semilla como línea tenue.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    x = np.array([h["trained_episodes"] for h in history], dtype=float)
    mean = np.array([h["mean"] for h in history], dtype=float)
    std = np.array([h["std"] for h in history], dtype=float)

    line, = ax.plot(x, mean, marker="o", label=label, color=color, zorder=3)
    c = line.get_color()
    ax.fill_between(x, mean - std, mean + std, alpha=0.20, color=c, zorder=1)

    if show_seeds and history and "seed_means" in history[0]:
        n = len(history[0]["seed_means"])
        for j in range(n):
            ys = [h["seed_means"][j] for h in history]
            ax.plot(x, ys, color=c, alpha=0.30, lw=1, zorder=2)

    n_seeds = history[0].get("n_seeds", "?") if history else "?"
    ax.set_xlabel("Episodios de entrenamiento")
    ax.set_ylabel("Recompensa de test (media entre semillas)")
    ax.set_title(f"Curva de aprendizaje — banda = ± std entre semillas (n={n_seeds}); "
                 "líneas tenues = cada semilla")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_success_rate_multiseed(history, label=None, ax=None):
    """Tasa de éxito media entre semillas, con banda ± std entre semillas."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    x = np.array([h["trained_episodes"] for h in history], dtype=float)
    mean = np.array([h["success_rate"] * 100 for h in history], dtype=float)
    std = np.array([h.get("success_std", 0.0) * 100 for h in history], dtype=float)
    line, = ax.plot(x, mean, marker="s", label=label)
    c = line.get_color()
    ax.fill_between(x, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100),
                    alpha=0.20, color=c)
    ax.set_xlabel("Episodios de entrenamiento")
    ax.set_ylabel("Tasa de éxito (%)")
    ax.set_ylim(-2, 102)
    ax.set_title("Tasa de éxito en test (media ± std entre semillas)")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_multiseed_report(result, outdir=None, show=True):
    """Figura resumen multi-semilla de una config: curva de aprendizaje (banda
    entre semillas + cada semilla) + tasa de éxito. Guarda .png si outdir."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_learning_curve_multiseed(result["history"], label=result["config_name"], ax=axes[0])
    plot_success_rate_multiseed(result["history"], label=result["config_name"], ax=axes[1])
    fig.suptitle(f"Reporte multi-semilla — {result['config_name']}", fontsize=14)
    fig.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{result['config_name']}_multiseed.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print("Guardado:", path)
    if show:
        plt.show()
    return fig
