"""Discretizaciones y set de configuraciones, AGRUPADAS POR DECISIÓN.

En vez de un único barrido sin estructura, se varía UN factor por vez alrededor
de la configuración ganadora ("base"). Así, la gráfica comparativa de cada grupo
justifica por sí sola una decisión, sin depender de la tabla del barrido previo.

Ejes de decisión (grupos):
  - posicion_bins : 10, 20(base), 30, 100  -> cuántos bins de posición
  - velocidad_disc: uniforme vs no uniforme -> cómo discretizar la velocidad
  - acciones      : 2, 3(base), 5           -> cuántas acciones discretas
  - gamma         : 0.9 vs 0.999(base)       -> factor de descuento
  - alpha         : 0.05, 0.1(base), 0.2     -> tasa de aprendizaje
  - epsilon       : fijo 0.9(base) vs decay  -> exploración

La config "base" pertenece a TODOS los grupos: es el pivote de comparación.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Discretizaciones
# --------------------------------------------------------------------------- #
def uniform_pos(n_bins=20):
    return np.linspace(-1.2, 0.6, n_bins)


def uniform_vel(n_bins=20):
    return np.linspace(-0.07, 0.07, n_bins)


def nonuniform_vel():
    """Bins de velocidad concentrados cerca de 0 (para detectar cambios de
    dirección). ~20 bordes, sin duplicados en el centro."""
    return np.concatenate([
        np.linspace(-0.07, -0.01, 5),
        np.linspace(-0.01, 0.01, 10)[1:-1],
        np.linspace(0.01, 0.07, 5),
    ])


def make_get_state(x_space, vel_space):
    """Fabrica get_state a partir de los bordes de discretización."""
    def get_state(obs):
        x, vel = obs
        x_bin = np.digitize(x, x_space)
        vel_bin = np.digitize(vel, vel_space)
        return x_bin, vel_bin
    return get_state


def make_actions(n=3):
    return list(np.linspace(-1, 1, n))


# --------------------------------------------------------------------------- #
# Etiquetas legibles de cada grupo (para títulos de las figuras)
# --------------------------------------------------------------------------- #
GROUPS = {
    "posicion_bins":  "Cantidad de bins de posición",
    "velocidad_disc": "Discretización de la velocidad (uniforme vs no uniforme)",
    "acciones":       "Cantidad de acciones discretas",
    "gamma":          "Factor de descuento gamma",
    "alpha":          "Tasa de aprendizaje alpha",
    "epsilon":        "Estrategia de exploración (epsilon fijo vs decay)",
}


# --------------------------------------------------------------------------- #
# Set de configuraciones
# --------------------------------------------------------------------------- #
def build_configs():
    """Lista de configs. Cada una: name, groups, x_space, vel_space, actions,
    hyper (alpha, gamma, epsilon, epsilon_min, epsilon_decay)."""
    base_hyper = dict(alpha=0.1, gamma=0.999, epsilon=0.9,
                      epsilon_min=0.05, epsilon_decay=1.0)
    vel_nu = nonuniform_vel()
    cfgs = []

    # ---- BASE (ganadora) — pivote de TODOS los grupos -------------------- #
    cfgs.append(dict(
        name="base_posU20_velNU_3acc",
        groups=["posicion_bins", "velocidad_disc", "acciones",
                "gamma", "alpha", "epsilon"],
        x_space=uniform_pos(20), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(base_hyper),
    ))

    # ---- GRUPO: bins de posición (vel no uniforme, 3 acc) ---------------- #
    cfgs.append(dict(name="pos_uniforme_10", groups=["posicion_bins"],
        x_space=uniform_pos(10), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(base_hyper)))
    cfgs.append(dict(name="pos_uniforme_30", groups=["posicion_bins"],
        x_space=uniform_pos(30), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(base_hyper)))
    cfgs.append(dict(name="pos_uniforme_100", groups=["posicion_bins"],
        x_space=uniform_pos(100), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(base_hyper)))  # demasiados bins -> tabla muy esparsa (lento)

    # ---- GRUPO: discretización de velocidad ------------------------------ #
    cfgs.append(dict(name="vel_uniforme_20", groups=["velocidad_disc"],
        x_space=uniform_pos(20), vel_space=uniform_vel(20), actions=make_actions(3),
        hyper=dict(base_hyper)))

    # ---- GRUPO: cantidad de acciones ------------------------------------- #
    cfgs.append(dict(name="acciones_2", groups=["acciones"],
        x_space=uniform_pos(20), vel_space=vel_nu, actions=make_actions(2),
        hyper=dict(base_hyper)))
    cfgs.append(dict(name="acciones_5", groups=["acciones"],
        x_space=uniform_pos(20), vel_space=vel_nu, actions=make_actions(5),
        hyper=dict(base_hyper)))

    # ---- GRUPO: gamma ---------------------------------------------------- #
    h = dict(base_hyper); h["gamma"] = 0.9
    cfgs.append(dict(name="gamma_0.9", groups=["gamma"],
        x_space=uniform_pos(20), vel_space=vel_nu, actions=make_actions(3), hyper=h))

    # ---- GRUPO: alpha ---------------------------------------------------- #
    h = dict(base_hyper); h["alpha"] = 0.05
    cfgs.append(dict(name="alpha_0.05", groups=["alpha"],
        x_space=uniform_pos(20), vel_space=vel_nu, actions=make_actions(3), hyper=h))
    h = dict(base_hyper); h["alpha"] = 0.2
    cfgs.append(dict(name="alpha_0.2", groups=["alpha"],
        x_space=uniform_pos(20), vel_space=vel_nu, actions=make_actions(3), hyper=h))

    # ---- GRUPO: epsilon (decay) ------------------------------------------ #
    h = dict(base_hyper); h["epsilon"] = 1.0; h["epsilon_decay"] = 0.9995; h["epsilon_min"] = 0.05
    cfgs.append(dict(name="epsilon_decay_0.9995", groups=["epsilon"],
        x_space=uniform_pos(20), vel_space=vel_nu, actions=make_actions(3), hyper=h))

    return cfgs


def configs_in_group(group_key, cfgs=None):
    """Configs que pertenecen a un grupo (la base aparece en todos)."""
    cfgs = cfgs or build_configs()
    return [c for c in cfgs if group_key in c["groups"]]
