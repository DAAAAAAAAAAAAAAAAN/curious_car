from dqn import env


def render_environment(start, actions, seed):
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