class Config:
    """Class for argument parsing and default arguments."""
    device = "cpu"
    seed = 43

    render = False
    curious = True
    save_to_disk = True

    batch_size = 64
    replay_memory_size = 10000
    num_experiments = 1
    num_episodes = 1000
    discount_factor = 0.99

    # Q-learning working learning rates
    # lr_curiosity_model = 5e-5
    # lr_q_model = 5e-4

    ## Optimized for INtrinsic rewards
    # Learning
    lr_curiosity_model = 5e-6
    lr_q_model = 1e-4
    # Networks
    num_hidden_q_model = 100
    num_hidden_curiosity_model = 10



    ## Optimized for EXtrinsic rewards
    # Learning
    lr_curiosity_model = 0
    lr_q_model = 1e-4
    # Networks
    num_hidden_q_model = 100
    num_hidden_curiosity_model = 1
