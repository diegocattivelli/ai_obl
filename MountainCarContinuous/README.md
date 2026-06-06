# Mountain Car Continuous — Q-Learning (Proyecto LOST)

Agente Q-Learning tabular para `MountainCarContinuous-v0` (Gymnasium), con
discretización de estados y acciones, exploración de hiperparámetros y
metodología de **entrenamiento/evaluación iterativa** (curvas de aprendizaje).

> Estado actual: **v2 cerrada**. Próximo: v2.1 (multi-semilla) → init de tabla /
> más episodios → Dyna-Q. Ver "Roadmap" al final.

---

## Metodología (lo importante)

En vez de entrenar cada configuración entera y testear una sola vez al final, se
**intercala** entrenamiento y evaluación:

```
[test base] → train 1000 → test 100 → train 1000 → test 100 → … (×10 = 10 000 episodios)
```

En cada checkpoint se registran, sobre los episodios de test (greedy, sin
explorar): media, desvío, min, max, percentiles 25/50/75, **tasa de éxito** y
pasos promedio. Con eso se grafican **curvas de aprendizaje con bandas de rango**,
que revelan inestabilidad / mesetas / varianza que el número final esconde.

La tabla Q se guarda en `.pkl` en cada checkpoint (modelo computado + reanudable).

## Configuraciones (un factor por vez)

Se varía un solo factor por vez alrededor de la config ganadora ("base"), de modo
que la comparativa de cada grupo justifica una decisión por sí sola.

| Grupo | Configs | Conclusión |
|---|---|---|
| Bins de posición  | 10, **20**, 30, 100 | 20–30 es el punto dulce; 10 inestable, 100 aprende pero lentísimo |
| Discretización velocidad | uniforme vs **no uniforme** | no uniforme (bins cerca de v≈0) aprende antes y más estable |
| Acciones          | 2, **3**, 5 | menos acciones = más rápido/estable; 5 muy inestable; 3 es el balance |
| Gamma             | 0.9 vs **0.999** | con γ=0.9 **no aprende** (recompensa diferida); 0.999 obligatorio |
| Alpha             | 0.05, **0.1**, 0.2 | 0.1 equilibra: 0.05 lento, 0.2 oscila |
| Epsilon           | **0.9 fijo** vs decay | recompensa final equivalente; el decay converge más estable |

**Config ganadora (base):** posición uniforme 20, velocidad no uniforme (~20),
3 acciones `[-1,0,1]`, α=0.1, γ=0.999, ε=0.9 fijo.

**Config final v2** (`final_v2_pos30_velNU_3acc_epsdecay_slow`): combina los dos
hallazgos de mayor estabilidad → posición **30** + velocidad no uniforme + 3
acciones + α=0.1 + γ=0.999 + **ε con decay lento** (1.0 → 0.1, decay 0.9998).
Aprende a los ~2000 episodios y se mantiene estable al 100% de éxito.

> **Hallazgo (interacción posición × exploración).** El primer intento naive
> (`final_v2_pos30_velNU_3acc_epsdecay`, decay 0.9995, ε_min 0.05) **NO aprende**:
> la tabla más grande de pos30 necesita más exploración, pero ese decay —calibrado
> para pos20— la recorta antes de descubrir la meta, y el agente queda clavado en
> recompensa 0. Se corrige estirando el decay (0.9998, ε_min 0.1). Muestra que los
> óptimos de cada eje por separado no necesariamente se combinan. Ambas configs se
> conservan en `configs.py` (`build_final_config_naive` y `build_final_config`).

## Archivos

| Archivo | Rol |
|---|---|
| `q_learning_agent.py` | Agente: entrenamiento **incremental** (Q y ε persisten), `evaluate` con tasa de éxito, `save`/`load` (.pkl). |
| `train_eval_loop.py` | Harness del loop train/test: `run_train_eval`, `resume_train_eval`. Genera `.pkl`, `_metrics.csv`, `_history.json`. |
| `configs.py` | Discretizaciones + 11 configs agrupadas + config final (`build_final_config`). |
| `plotting.py` | Curva de aprendizaje (banda std/minmax/iqr), tasa de éxito, recompensa de train, comparación. |
| `make_plots.py` | Comparativa **por grupo** desde los `_history.json` (sin gymnasium; regenera figuras sin reentrenar). |
| `run_experiments.py` | Runner de terminal. |
| `continuous_mountain_car.ipynb` | Notebook orquestador. |
| `README_metodologia.md` | Detalle extendido de la metodología. |

## Cómo correr

```bash
poetry run python run_experiments.py                 # 11 configs de grupos + 2 finales
poetry run python run_experiments.py --group acciones
poetry run python run_experiments.py --only final_v2_pos30_velNU_3acc_epsdecay_slow  # final
poetry run python run_experiments.py --only final_v2_pos30_velNU_3acc_epsdecay        # naive (no aprende)
poetry run python run_experiments.py --cycles 5 --train 200 --eval 20   # prueba rápida
poetry run python make_plots.py                      # regenerar comparativas por grupo
poetry run python run_experiments.py --resume checkpoints/<config>_ep10000.pkl --cycles 5
```

Salidas en `checkpoints/`: `<config>_ep<N>.pkl`, `<config>_metrics.csv`,
`<config>_history.json`, `figs/<config>_reporte.png`, `figs/grupo_<grupo>.png`.

> Nota: cada config tarda varios minutos (`pos_100` es la más lenta). Se puede
> correr grupo por grupo en distintas sesiones; `make_plots.py` arma las
> comparativas con los historiales que haya.

## Decisiones de diseño

- **Evaluación con semilla fija** (`eval_seed=0`): mismos estados iniciales en cada
  checkpoint → curvas comparables.
- **Éxito** = el episodio termina en la meta (`terminated`), no por límite de pasos.
- La recompensa de **train** se ve baja porque se entrena con ε-greedy (mucha
  exploración); el rendimiento real se mide aparte en modo greedy (test).
- Entrenamiento con `seed=0` (única semilla por ahora — ver v2.1).

---

## Roadmap

- [x] **v2** — loop train/test, 11 configs agrupadas, comparativas por grupo, config final.
- [ ] **v2.1 — Multi-semilla.** Repetir entrenos con varias semillas, agregar banda
  entre semillas a las curvas, re-interpretar y definir nueva config final robusta.
  (Verifica si el pozo de la base ~ep.7000 es propio de seed=0 o sistemático.)
- [ ] **Init de tabla Q / más episodios.** Inicialización optimista; extender
  `pos_100` con `--resume` para confirmar que converge.
- [ ] **Dyna-Q** (obligatorio, punto 4). Reusa este harness sobre la versión final
  del Q-learning normal; variar nº de pasos de planning (0, 5, 50) como grupo.

## Punto de retorno (para retomar)

Si volvés a este commit: la v2 está completa y reproducible. Para seguir,
arrancá por v2.1 (multi-semilla) — el agente ya acepta `seed`; falta orquestar
varias corridas y promediar/bandear en `plotting.py`. El ejercicio 2 (MATE /
Isolation) ya está hecho aparte.
