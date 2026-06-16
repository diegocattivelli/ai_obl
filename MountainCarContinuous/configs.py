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

Además se definen DOS configs finales (groups=[]) que combinan los hallazgos:
  - build_final_config_naive(): pos30 + decay rápido -> NO aprende (documentado).
  - build_final_config(): pos30 + decay lento -> versión final corregida.
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


def nonuniform_pos():
    """Bins de posición concentrados en la zona que el auto realmente recorre:
    el valle (~-0.5, donde arranca y cambia de dirección) y la subida hacia la
    meta (~0.45). El extremo izquierdo (-1.2) casi nunca se visita, así que se le
    dan pocos bins. ~20 bordes en total, para ser comparable con uniform_pos(20)."""
    return np.concatenate([
        np.linspace(-1.2, -0.6, 3),       # extremo izquierdo: pocos bins
        np.linspace(-0.6, 0.3, 13)[1:],   # valle / zona activa: densa
        np.linspace(0.3, 0.6, 6)[1:],     # subida a la meta: densa
    ])


def nonuniform_pos_left():
    """Variante que concentra los bins en la IZQUIERDA (la colina por la que el
    auto sube para juntar impulso) y deja pocos hacia la meta. Es la hipótesis
    corregida: si robarle resolución a la izquierda fue lo que hizo perder a
    `nonuniform_pos`, dársela debería ayudar. ~20 bordes, comparable a uniform_pos(20)."""
    return np.concatenate([
        np.linspace(-1.2, -0.3, 14)[:-1],  # izquierda + valle: densa (13 bordes)
        np.linspace(-0.3, 0.6, 7),         # derecha hacia la meta: rala (7 bordes)
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
    "posicion_disc":  "Discretización de la posición (uniforme vs no uniforme)",
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
        groups=["posicion_bins", "posicion_disc", "velocidad_disc", "acciones",
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

    # ---- GRUPO: discretización de posición (forma: unif vs no unif) ------ #
    # La base (pos uniforme 20) es la referencia uniforme; acá la contraparte
    # no uniforme (mismos ~20 bins, concentrados en el valle y cerca de la meta).
    cfgs.append(dict(name="pos_nouniforme", groups=["posicion_disc"],
        x_space=nonuniform_pos(), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(base_hyper)))
    # concentrada en la izquierda (hipótesis corregida: más bins en la colina de impulso)
    cfgs.append(dict(name="pos_nouniforme_izq", groups=["posicion_disc"],
        x_space=nonuniform_pos_left(), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(base_hyper)))

    # ---- GRUPO: discretización de velocidad (forma: unif vs no unif) ----- #
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
    # epsilon con decay LENTO (cambio de un solo factor sobre la base: solo epsilon)
    h = dict(base_hyper); h["epsilon"] = 1.0; h["epsilon_decay"] = 0.9998; h["epsilon_min"] = 0.1
    cfgs.append(dict(name="epsilon_decay_slow", groups=["epsilon"],
        x_space=uniform_pos(20), vel_space=vel_nu, actions=make_actions(3), hyper=h))

    # ---- COMBINACIONES PARA VALIDACIÓN (groups=[] -> fuera de las comparativas) ----
    # No son OFAT: combinan ganadores para construir/estresar la config final.
    h_slow = dict(alpha=0.1, gamma=0.999, epsilon=1.0, epsilon_decay=0.9998, epsilon_min=0.1)
    # (1) candidata a final: posición no uniforme izquierda + decay lento
    #     (los dos ingredientes de aprendizaje rápido temprano -> ¿se suman?)
    cfgs.append(dict(name="pos_nouniforme_izq_epsdecay_slow", groups=[],
        x_space=nonuniform_pos_left(), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(h_slow)))
    # (2) confirmación de mecanismo: tabla grande (pos100) + decay lento
    #     (si el 100 era lento por falta de visitas, más exploración debería acelerarlo)
    cfgs.append(dict(name="pos100_epsdecay_slow", groups=[],
        x_space=uniform_pos(100), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(h_slow)))

    # Versiones con decay RÁPIDO (0.9995): es el decay que en el OFAT empató con la
    # base (el lento se degradaba al final), así que es el correcto para combinar.
    h_fast = dict(alpha=0.1, gamma=0.999, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05)
    cfgs.append(dict(name="pos_nouniforme_izq_epsdecay_fast", groups=[],
        x_space=nonuniform_pos_left(), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(h_fast)))
    cfgs.append(dict(name="pos100_epsdecay_fast", groups=[],
        x_space=uniform_pos(100), vel_space=vel_nu, actions=make_actions(3),
        hyper=dict(h_fast)))

    # ---- CONFIGS FINALES v2 (groups=[] -> fuera de las comparativas) ----- #
    cfgs.append(build_final_config_naive())  # documentado: NO aprende (interacción)
    cfgs.append(build_final_config())        # corregido: decay más lento

    return cfgs


def build_final_config_naive():
    """Intento NAIVE de combinar pos30 + decay (resultado documentado).

    NO APRENDE: la tabla más grande (pos 30) necesita más exploración, pero el
    decay rápido (0.9995, ε_min 0.05) la recorta antes de descubrir la meta.
    Se conserva como resultado negativo: muestra que los óptimos de cada eje por
    separado no necesariamente se combinan (interacción posición × exploración).
    """
    return dict(
        name="final_v2_pos30_velNU_3acc_epsdecay",
        groups=[],
        x_space=uniform_pos(30),
        vel_space=nonuniform_vel(),
        actions=make_actions(3),
        hyper=dict(alpha=0.1, gamma=0.999,
                   epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05),
    )


def build_final_config():
    """Configuración final CORREGIDA de la v2.

    pos30 + velocidad no uniforme + 3 acciones + alpha 0.1 + gamma 0.999 +
    epsilon con decay LENTO (0.9998, ε_min 0.1). El decay más lento mantiene
    exploración suficiente para llenar la tabla más grande, evitando el fallo
    del intento naive.
    """
    return dict(
        name="final_v2_pos30_velNU_3acc_epsdecay_slow",
        groups=[],
        x_space=uniform_pos(30),
        vel_space=nonuniform_vel(),
        actions=make_actions(3),
        hyper=dict(alpha=0.1, gamma=0.999,
                   epsilon=1.0, epsilon_decay=0.9998, epsilon_min=0.1),
    )


def configs_in_group(group_key, cfgs=None):
    """Configs que pertenecen a un grupo (la base aparece en todos)."""
    cfgs = cfgs or build_configs()
    return [c for c in cfgs if group_key in c["groups"]]
