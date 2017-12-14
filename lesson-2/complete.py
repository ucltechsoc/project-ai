import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import random


class Gravity:
    def __init__(self, size):
        self.size = size
        block_y = 0
        block_x = random.randint(0, self.size - 1)
        basket_x = random.randint(1, self.size - 2)
        self.state = [block_y, block_x, basket_x]

    def observe(self):
        canvas = [0] * self.size**2
        canvas[self.state[0] * self.size + self.state[1]] = 1
        canvas[(self.size - 1) * self.size + self.state[2] - 1] = 1
        canvas[(self.size - 1) * self.size + self.state[2] + 0] = 1
        canvas[(self.size - 1) * self.size + self.state[2] + 1] = 1
        return np.array(canvas).reshape((1, -1))

    def act(self, action):
        block_y, block_x, basket_x = self.state

        basket_x += (int(action) - 1)
        basket_x = max(1, basket_x)
        basket_x = min(self.size - 2, basket_x)

        block_y += 1

        self.state = [block_y, block_x, basket_x]

        reward = 0
        if block_y == self.size - 1:
            if abs(block_x - basket_x) <= 1:
                reward = 1
            else:
                reward = -1

        game_over = block_y == self.size - 1

        return self.observe(), reward, game_over

    def reset(self):
        self.__init__(self.size)


class ExperienceReplay(object):
    def __init__(self, max_memory, discount):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        self.memory.append([states, game_over])

        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size):
        len_memory = len(self.memory)

        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])

            if game_over:
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets

if __name__ == "__main__":
    # Constants
    GRID_DIM = 10

    EPSILON = .1
    LEARNING_RATE = .2
    LOSS_FUNCTION = "mse"

    HIDDEN_LAYER1_SIZE = 100
    HIDDEN_LAYER1_ACTIVATION = "relu"

    HIDDEN_LAYER2_SIZE = 100
    HIDDEN_LAYER2_ACTIVATION = "relu"

    BATCH_SIZE = 50
    EPOCHS = 1000
    MAX_MEMORY = 500
    DISCOUNT = .9

    # Model Setup
    model = Sequential()
    model.add(Dense(HIDDEN_LAYER1_SIZE, input_shape=(GRID_DIM**2,), activation=HIDDEN_LAYER1_ACTIVATION))
    model.add(Dense(HIDDEN_LAYER2_SIZE, activation=HIDDEN_LAYER2_ACTIVATION))
    model.add(Dense(3))
    model.compile(sgd(lr=LEARNING_RATE), LOSS_FUNCTION)

    # Environment Setup
    env = Gravity(GRID_DIM)

    # Experience Setup
    replay = ExperienceReplay(MAX_MEMORY, DISCOUNT)

    # Run Model
    win_cnt = 0
    for e in range(EPOCHS):
        loss = 0.
        env.reset()
        game_over = False
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t

            if random.random() <= EPSILON:
                action = random.randint(0, 2)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            input_t, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1

            replay.remember([input_tm1, action, reward, input_t], game_over)
            inputs, targets = replay.get_batch(model, batch_size=BATCH_SIZE)
            loss += model.train_on_batch(inputs, targets)

        print("Epoch {:06d}/{:06d} | Loss {:.4f} | Win count {}".format(e, EPOCHS, loss, win_cnt))

    # Save Model
    model.save_weights("trained.h5", overwrite=True)

    with open("trained.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
