import json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from .lesson import Gravity, GRID_DIM


if __name__ == "__main__":
    with open("trained.json", "r") as file:
        model = model_from_json(json.load(file))
        model.load_weights("trained.h5")
        model.compile("sgd", "mse")

    # Environment Setup
    env = Gravity(GRID_DIM)
    c = 0

    for e in range(10):
        loss = 0.
        env.reset()
        game_over = False
        input_t = env.observe()

        plt.imshow(input_t.reshape((GRID_DIM,)*2), interpolation='none', cmap='gray')
        plt.savefig("%03d.png" % c)
        c += 1

        while not game_over:
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            plt.imshow(input_t.reshape((GRID_DIM,)*2), interpolation='none', cmap='gray')
            plt.savefig("%03d.png" % c)
            c += 1
