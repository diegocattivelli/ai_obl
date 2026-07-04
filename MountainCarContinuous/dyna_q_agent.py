import os
import pickle
import random

import numpy as np


class DynaQAgent:
    """Agente Dyna-Q tabular para Mountain Car Continuous (discretizado).

    Es Q-Learning + un modelo del entorno + planificacion. Respecto a
    QLearningAgent agrega dos cosas por cada paso real:
      (e) guarda la transicion observada en un modelo (entorno deterministico),
      (f) hace `n_planning_steps` updates extra con experiencia simulada del modelo.

    Con n_planning_steps = 0 se reduce EXACTAMENTE a Q-Learning, lo que lo hace
    un baseline natural.

    Mantiene la MISMA interfaz que QLearningAgent (train incremental con Q y
    epsilon persistentes, evaluate con recompensa/exito/pasos, save/load) para
    poder enchufarlo en train_eval_loop.py y sacar las mismas curvas.
    """

    def __init__(self, n_pos, n_vel, n_actions, alpha=0.1, gamma=0.999,
                 epsilon=0.9, epsilon_min=0.05, epsilon_decay=1.0,
                 seed=None, n_planning_steps=5):
        self.n_pos = n_pos
        self.n_vel = n_vel
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_planning_steps = n_planning_steps
        self.q = np.zeros((n_pos, n_vel, n_actions))
        # model[(state, action_index)] = (reward, next_state, terminated)
        self.model = {}
        self._keys = []            # lista de claves del modelo (muestreo O(1))
        self.trained_episodes = 0
        self.trained_steps = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------ #
    def next_action(self, state):
        """Accion greedy: indice que maximiza Q para el estado dado."""
        return int(np.argmax(self.q[state]))

    def _epsilon_greedy_policy(self, state, epsilon):
        if np.random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q[state]))

    def _q_update(self, state, action_index, reward, next_state, terminated):
        """Update de Q-Learning (sirve para experiencia real y simulada)."""
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[next_state])
        self.q[state][action_index] += self.alpha * (
            target - self.q[state][action_index])

    # ------------------------------------------------------------------ #
    def train(self, env, get_state, actions, episodes):
        """Entrena `episodes` episodios CONTINUANDO el estado actual (Dyna-Q).
        Retorna {'rewards': [...], 'steps': [...], 'successes': [...]}."""
        rewards, steps_list, successes = [], [], []
        for _ in range(episodes):
            obs, _ = env.reset()
            current_state = get_state(obs)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            reached_goal = False
            while not done:
                action_index = self._epsilon_greedy_policy(current_state, self.epsilon)
                obs, reward, terminated, truncated, _ = env.step(np.array([actions[action_index]]))
                done = terminated or truncated
                next_state = get_state(obs)

                # (d) Direct RL: update con experiencia real
                self._q_update(current_state, action_index, reward, next_state, terminated)

                # (e) actualizar el modelo (entorno deterministico)
                key = (current_state, action_index)
                if key not in self.model:
                    self._keys.append(key)
                self.model[key] = (reward, next_state, terminated)

                # (f) planificacion: n updates con experiencia simulada
                for _ in range(self.n_planning_steps):
                    if not self._keys:
                        break
                    s, a = random.choice(self._keys)
                    r, s_next, term = self.model[(s, a)]
                    self._q_update(s, a, r, s_next, term)

                if terminated:
                    reached_goal = True
                current_state = next_state
                ep_reward += reward
                ep_steps += 1

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.trained_episodes += 1
            self.trained_steps += ep_steps
            rewards.append(ep_reward)
            steps_list.append(ep_steps)
            successes.append(reached_goal)
        return {"rewards": rewards, "steps": steps_list, "successes": successes}

    # ------------------------------------------------------------------ #
    def evaluate(self, env, get_state, actions, episodes=100, seed=None):
        """Evalua greedy `episodes` episodios (identico a QLearningAgent).
        Si `seed` no es None, evalua los mismos inicios en cada checkpoint."""
        rewards, steps_list, successes = [], [], []
        for i in range(episodes):
            if seed is not None:
                obs, _ = env.reset(seed=seed + i)
            else:
                obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            reached_goal = False
            while not done:
                state = get_state(obs)
                action_idx = self.next_action(state)
                obs, reward, terminated, truncated, _ = env.step(np.array([actions[action_idx]]))
                done = terminated or truncated
                if terminated:
                    reached_goal = True
                ep_reward += reward
                ep_steps += 1
            rewards.append(ep_reward)
            steps_list.append(ep_steps)
            successes.append(reached_goal)
        return {"rewards": rewards, "steps": steps_list, "successes": successes}

    # ------------------------------------------------------------------ #
    def save(self, path, extra=None):
        """Guarda tabla Q + modelo + metadata en un .pkl (reanudable)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "q": self.q, "n_pos": self.n_pos, "n_vel": self.n_vel,
            "n_actions": self.n_actions, "alpha": self.alpha, "gamma": self.gamma,
            "epsilon": self.epsilon, "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min, "epsilon_decay": self.epsilon_decay,
            "n_planning_steps": self.n_planning_steps,
            "model": self.model,
            "trained_episodes": self.trained_episodes,
            "trained_steps": self.trained_steps, "extra": extra or {},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    @classmethod
    def load(cls, path):
        """Carga un agente desde un .pkl. Devuelve (agent, extra)."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        agent = cls(payload["n_pos"], payload["n_vel"], payload["n_actions"],
                    alpha=payload["alpha"], gamma=payload["gamma"],
                    epsilon=payload["epsilon"], epsilon_min=payload["epsilon_min"],
                    epsilon_decay=payload["epsilon_decay"],
                    n_planning_steps=payload.get("n_planning_steps", 5))
        agent.q = payload["q"]
        agent.epsilon_start = payload.get("epsilon_start", payload["epsilon"])
        agent.model = payload.get("model", {})
        agent._keys = list(agent.model.keys())
        agent.trained_episodes = payload.get("trained_episodes", 0)
        agent.trained_steps = payload.get("trained_steps", 0)
        return agent, payload.get("extra", {})
