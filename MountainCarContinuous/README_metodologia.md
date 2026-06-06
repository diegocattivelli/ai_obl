# Metodología train/test iterativo — Q-Learning (LOST)

## Qué cambió respecto a la versión anterior

Antes: por cada configuración se entrenaba todo (p. ej. 10 000 episodios) y se
testeaba **una sola vez** al final (10 episodios), guardando solo el promedio.
Eso muestra un único número, no **cómo aprende** el agente.

Ahora se intercala entrenamiento y evaluación en un loop:

```
[test base] → train 1000 → test 100 → train 1000 → test 100 → … (×10)
```

En cada checkpoint se registran, sobre los episodios de test, estadísticas
completas: **media, desvío, min, max, percentiles 25/50/75, tasa de éxito y
pasos promedio**. Con eso se grafican **curvas de aprendizaje con bandas de
rango**, que pueden revelar comportamientos (inestabilidad, mesetas, varianza
alta) que el número final esconde.

La tabla Q se guarda como `.pkl` en cada checkpoint: permite **reanudar** el
entrenamiento y es el *modelo computado* exigido en la entrega.

## Configuraciones agrupadas por decisión

En vez de un barrido sin estructura, se varía **un factor por vez** alrededor de
la configuración ganadora ("base"). Cada grupo produce **una gráfica comparativa
que justifica una decisión por sí sola**, sin depender de la tabla del barrido
previo. Son 11 configuraciones en 6 grupos:

| Grupo | Configs | Decisión que justifica |
|---|---|---|
| `posicion_bins`  | 10, **20**, 30, 100 | cuántos bins de posición (10 pocos, 100 demasiados) |
| `velocidad_disc` | uniforme vs **no uniforme** | concentrar bins de velocidad cerca de 0 |
| `acciones`       | 2, **3**, 5 | cuántas acciones discretas |
| `gamma`          | 0.9 vs **0.999** | factor de descuento |
| `alpha`          | 0.05, **0.1**, 0.2 | tasa de aprendizaje |
| `epsilon`        | **fijo 0.9** vs decay 0.9995 | estrategia de exploración |

(la **base** en negrita es el pivote y aparece en todos los grupos)

## Archivos

| Archivo | Rol |
|---|---|
| `q_learning_agent.py` | Agente con entrenamiento **incremental** (tabla Q y epsilon persisten entre llamadas), evaluación con tasa de éxito, y `save`/`load` (.pkl). |
| `train_eval_loop.py` | Harness del loop train/test: `run_train_eval` y `resume_train_eval`. Genera `.pkl`, `_metrics.csv` y `_history.json`. |
| `configs.py` | Discretizaciones + las 11 configs agrupadas (`build_configs`, `configs_in_group`, `GROUPS`). |
| `plotting.py` | Curva de aprendizaje (banda `std`/`minmax`/`iqr`), tasa de éxito, recompensa de train suavizada, comparación. |
| `make_plots.py` | Comparativa **por grupo** desde los `_history.json` (sin gymnasium; regenera figuras sin reentrenar). |
| `run_experiments.py` | Corre el set desde la terminal. |
| `continuous_mountain_car.ipynb` | Notebook que orquesta todo. |

## Cómo correrlo

Con el entorno de Poetry:

```bash
poetry run python run_experiments.py                      # las 11 configs
poetry run python run_experiments.py --group acciones     # solo un grupo
poetry run python run_experiments.py --only base_posU20_velNU_3acc
poetry run python run_experiments.py --cycles 5 --train 200 --eval 20   # prueba rápida
poetry run python run_experiments.py --resume checkpoints/base_posU20_velNU_3acc_ep10000.pkl --cycles 5
poetry run python make_plots.py                           # regenerar comparativas por grupo
```

Como cada config tarda varios minutos, podés correr **grupo por grupo** en
distintas sesiones; `make_plots.py` arma las comparativas con los historiales
que haya disponibles.

### Salidas (en `checkpoints/`)

- `<config>_ep<N>.pkl` — modelos computados (reanudables).
- `<config>_metrics.csv` — una fila por checkpoint (para tablas del informe).
- `<config>_history.json` — historial completo (recompensas crudas incluidas).
- `figs/<config>_reporte.png` — curva de aprendizaje + tasa de éxito + train.
- `figs/grupo_<grupo>.png` — comparativa por eje de decisión.

## Notas de diseño

- **Evaluación con semilla fija** (`eval_seed=0`): en cada checkpoint se evalúan
  los **mismos** estados iniciales (`env.reset(seed=eval_seed+i)`), lo que hace
  las curvas comparables. Para medir robustez ante inicios aleatorios, `eval_seed=None`.
- **Éxito** = el episodio termina en la meta (`terminated=True`), no por límite
  de pasos (`truncated`).
- **epsilon fijo 0.9** por defecto (`epsilon_decay=1.0`); el grupo `epsilon`
  compara contra una variante con decaimiento.
- El tiempo de cómputo por config es similar al anterior (misma cantidad de
  episodios de entrenamiento); el costo extra son las evaluaciones intermedias.
