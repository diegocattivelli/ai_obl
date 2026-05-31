import random
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:

    # el agente neecsita conocer la tabla de Q
    # tenemos que tener posicion, velocidad y accion
    def __init__(self, n_pos, n_vel, n_actions):
        """Inicializa la tabla Q con ceros."""
        self.q = np.zeros((n_pos, n_vel, n_actions))

    # nos llega el estado donde estamos, me quedo con la acción de mejor utilidad, que me maximiza
    def next_action(self, state):
        """Retorna el índice de la acción que maximiza Q para el estado dado."""
        action = np.argmax(self.q[state])
        return action
    

    def _epsilon_greedy_policy(self, state, epsilon):
        """Con probabilidad epsilon explora, con probabilidad 1-epsilon explota. Retorna el índice de la acción elegida."""
        explore = np.random.binomial(1, epsilon)

        if explore:
            action_index = random.randint(0, self.q.shape[2] - 1) # indice random entre 0 y numero de acciones discretas - 1

        else: # exploit
            action_index = np.argmax(self.q[state])
        
        return action_index

    def train_agent(self, env, get_state, actions, episodes=10000, epsilon=0.9, gamma=0.999, alpha=0.1, epsilon_min=0.05, epsilon_decay=0.9995):
        """Entrena el agente usando Q-Learning durante un número de episodios.En cada paso actualiza Q.
        Retorna la tabla Q entrenada y la lista de recompensas por episodio."""
        rewards = []
        for episode in range (episodes):
            obs, _ = env.reset()
            current_state = get_state(obs)
            done = False
            rewards_episode=0
            while not done:
                action_index = self._epsilon_greedy_policy(current_state, epsilon)
                obs, reward, terminated, truncated, _ = env.step(np.array([actions[action_index]]))
                done = terminated or truncated
                next_state = get_state(obs)
                if terminated:
                    target = reward
                else: # not terminated
                    target = reward + gamma * np.max(self.q[next_state])
                self.q[current_state][action_index] += alpha * (target - self.q[current_state][action_index])
                current_state = next_state
                rewards_episode += reward
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            rewards.append(rewards_episode)
        return self.q, rewards
    
    def test_agent(self, env, get_state, actions, episodes=10):
        """Evalúa el agente entrenado durante un número de episodios, eligiendo siempre la acción óptima según la tabla Q (no explora).
        Retorna la lista de recompensas por episodio."""
        rewards = []
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            rewards_episode = 0
            while not done:
                state = get_state(obs)
                action_idx = self.next_action(state)
                obs, reward, terminated, truncated, _ = env.step(np.array([actions[action_idx]]))
                done = terminated or truncated
                rewards_episode += reward
            rewards.append(rewards_episode)
        return rewards