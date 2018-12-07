class Config:
    """Class for argument parsing and default arguments."""
    device = "cpu"
    seed = 43

    # Learning
    lr_curiosity_model = 1e-4
    lr_q_model = 1e-4
    batch_size = 64
    replay_memory_size = 10000
    num_episodes = 500
    discount_factor = 0.97

    # Networks
    num_hidden_q_model = 200
    num_hidden_curiosity_model = 200

    render = True
    curious = True

