import json
import os
import time
from datetime import date

import torch


def render_environment(env, start, actions, seed):
    import time

    env.seed(seed)
    state = env.reset()

    assert (state == start).all(), (state, start)

    env.render()
    ep_length = 0

    for action in actions:
        # perform action
        _, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.05)

        ep_length += 1

    print('finished in', ep_length)
    env.close()  # Close the environ


def save_check_point(q_model, curiosity_model, config, **kwargs):
    path = "checkpoints/"
    if not os.path.isdir(path):
        os.mkdir(path)

    file = f"{date.today()}-{int(time.time())}"
    file_meta = file + ".json"
    file_q_model = file + "-q-model.data"
    file_curiosity_model = file + "-curiosity-model.data"

    torch.save(q_model.state_dict(),
               os.path.join(path, file_q_model))
    torch.save(curiosity_model.state_dict(),
               os.path.join(path, file_curiosity_model))

    with open(os.path.join(path, file_meta), "w") as f:
        f.write(json.dumps({
            **vars(config),
            'q_model': file_q_model,
            'curiosity_model': file_curiosity_model,
            **kwargs
        }, indent=4))
