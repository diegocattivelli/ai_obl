import random
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:

    # el agente neecsita conocer la tabla de Q
    # tenemos que tener posicion, velocidad y accion
    def __init__(self, n_pos, n_vel, n_actions):
        self.q = np.zeros((n_pos, n_vel, n_actions))

    # nos llega el estado donde estamos, me quedo con la acción de mejor utilidad, que me maximiza
    def next_action(self, state):
        action = np.argmax(self.q[state])
        return action
    

    def _epsilon_greedy_policy(self, state, epsilon):
        explore = np.random.binomial(1, epsilon)
        
        if explore:
            action_index = random.randint(0, self.q.shape[2] - 1) # indice random entre 0 y numero de acciones discretas - 1

        else: # exploit
            action_index = np.argmax(self.q[state])
        
        return action_index

    #entrenar el agente, mapear lo de la diapostiva de como cargar la tabla de Q
    # acá hacemos muchos episodios, ejecutar muchas veces nuestro agente 
    #devuelvo la tabla Q y las recompensas.
    def train_agent(self, env, get_state, actions, episodes=1000, epsilon=0.9, gamma=0.9, alpha=0.1):
        rewards = []
        for episode in range (episodes):
            obs, _ = env.reset()
            current_state = get_state(obs)
            done = False
            rewards_episode=0
            while not done:
                action_index = self._epsilon_greedy_policy(current_state, epsilon)
                obs, reward, done, _, _ = env.step(np.array([actions[action_index]]))
                next_state = get_state(obs)
                self.q[current_state][action_index] += alpha*(reward + gamma*np.max(self.q[next_state])-self.q[current_state][action_index])
                current_state = next_state
                rewards_episode += reward
            rewards.append(rewards_episode)
        return self.q, rewards
    
    #toma la accion máxima, a partir de la tabla de q va a pbtener la accion que maximiza la utilidad
    #depsues de entrenar al agente realizar un test en el cual el agente elije la accion optima basada en la tabla q entrenada
    def test_agent(self, env, get_state, actions, episodes=10):
        rewards = []
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            rewards_episode = 0
            while not done:
                state = get_state(obs)
                action_idx = self.next_action(state)
                obs, reward, done, _, _ = env.step(np.array([actions[action_idx]]))
                rewards_episode += reward
            rewards.append(rewards_episode)
        return rewards