import os
import pickle
import random

import numpy as np


class QLearningAgent:
    """Agente Q-Learning tabular para Mountain Car Continuous (discretizado).

    El estado de entrenamiento (tabla Q, epsilon actual, episodios entrenados)
    es PERSISTENTE en la instancia: permite llamar a `train` muchas veces y que
    el aprendizaje CONTINÚE donde quedó (habilita el loop train/test). `evaluate`
    devuelve recompensa, éxito y pasos por episodio. `save`/`load` serializan la
    tabla Q + metadata (modelo computado .pkl, reanudable).
    """

    def __init__(self, n_pos, n_vel, n_actions, alpha=0.1, gamma=0.999,
                 epsilon=0.9, epsilon_min=0.05, epsilon_decay=1.0, seed=None):
        self.n_pos = n_pos
        self.n_vel = n_vel
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = np.zeros((n_pos, n_vel, n_actions))
        self.trained_episodes = 0
        self.trained_steps = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------ #
    def next_action(self, state):
        """Acción greedy: índice que maximiza Q para el estado dado."""
        return int(np.argmax(self.q[state]))

    def _epsilon_greedy_policy(self, state, epsilon):
        if np.random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q[state]))

    # ------------------------------------------------------------------ #
    def train(self, env, get_state, actions, episodes):
        """Entrena `episodes` episodios CONTINUANDO el estado actual.
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
                if terminated:
                    target = reward
                    reached_goal = True
                else:
                    target = reward + self.gamma * np.max(self.q[next_state])
                self.q[current_state][action_index] += self.alpha * (
                    target - self.q[current_state][action_index])
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
        """Evalúa greedy `episodes` episodios. Si `seed` no es None, evalúa los
        mismos estados iniciales en cada checkpoint (curvas comparables).
        Retorna {'rewards': [...], 'steps': [...], 'successes': [...]}."""
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
        """Guarda la tabla Q + metadata en un .pkl para retomar luego."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "q": self.q, "n_pos": self.n_pos, "n_vel": self.n_vel,
            "n_actions": self.n_actions, "alpha": self.alpha, "gamma": self.gamma,
            "epsilon": self.epsilon, "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min, "epsilon_decay": self.epsilon_decay,
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
                    epsilon_decay=payload["epsilon_decay"])
        agent.q = payload["q"]
        agent.epsilon_start = payload.get("epsilon_start", payload["epsilon"])
        agent.trained_episodes = payload.get("trained_episodes", 0)
        agent.trained_steps = payload.get("trained_steps", 0)
        return agent, payload.get("extra", {})

    # ------------------------------------------------------------------ #
    # Compatibilidad hacia atrás con el notebook anterior
    # ------------------------------------------------------------------ #
    def train_agent(self, env, get_state, actions, episodes=10000, epsilon=None,
                    gamma=None, alpha=None, epsilon_min=None, epsilon_decay=None):
        """Wrapper compatible: entrena `episodes` y devuelve (Q, rewards)."""
        if epsilon is not None:
            self.epsilon = epsilon
        if gamma is not None:
            self.gamma = gamma
        if alpha is not None:
            self.alpha = alpha
        if epsilon_min is not None:
            self.epsilon_min = epsilon_min
        if epsilon_decay is not None:
            self.epsilon_decay = epsilon_decay
        out = self.train(env, get_state, actions, episodes)
        return self.q, out["rewards"]

    def test_agent(self, env, get_state, actions, episodes=10, seed=None):
        """Wrapper compatible: devuelve la lista de recompensas de evaluación."""
        return self.evaluate(env, get_state, actions, episodes, seed=seed)["rewards"]
