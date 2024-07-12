from datetime import datetime

""" Hyperparameters of Priority Learning!"""


class EnvParam:
    N_AGENTS = 128  # number of agents used in training
    WORLD_SIZE = (32, 32)
    OBSTACLE_PROB = (0.2, 0.21)


class TrainParam:
    lr = 3e-4
    GAMMA = 0.95  # discount factor
    max_grad_norm = 1
    N_ENVS = 32  # number of processes
    max_steps = int(3e7)  # maximum number of time steps used in training
    N_STEPS =EnvParam.N_AGENTS * 4   # number of time steps per process per data collection
    MAX_EPISODE_LEN = int(500) # make sure bigger than n_agents

    LOG_PERIOD = N_ENVS*N_STEPS*10 # per epoch
    SAVE_PERIOD = N_ENVS*N_STEPS*40  # per epoch
    
    CRITIC_MAX_VALUE = 1
    
    UPDATE_EPOCH = 1
    # specify the training algorithm
    # reinforce, ppo.
    ALGORITHM = "reinforce"
    
    baseline = None # old policy, expert
    
    # ppo algorithm
    EPS_CLIP = 0.2
    
    demo_weight = 0
    rl_weight = 1
    
    entropy_weight = 0.1
    
    load_pretrain = False
    pretrain_path = "./PPO_preTrained/MapfSippsEnv/PPO_MapfSippsEnv_0_59.pth"
    


class NetParameters:
    EMBEDDING_DIM = 128
    N_ENCODER_LAYERS = 1  # number of computation block
    N_HEAD = 8
    checkpoint_encoder = False



class SetupParameters:
    SEED = 1234
    USE_GPU_SAMPLE = False
    USE_GPU_TRAIN = True
    NUM_GPU = 1



all_args = {'N_AGENTS': EnvParam.N_AGENTS, 
            'WORLD_SIZE': EnvParam.WORLD_SIZE,
            'OBSTACLE_PROB': EnvParam.OBSTACLE_PROB,
            'lr': TrainParam.lr, 'GAMMA': TrainParam.GAMMA, 
            'MAX_GRAD_NORM': TrainParam.max_grad_norm,
            'N_ENVS': TrainParam.N_ENVS,
            'N_MAX_STEPS': TrainParam.max_steps,
            'N_STEPS': TrainParam.N_STEPS, 
            'N_LAYERS': NetParameters.N_ENCODER_LAYERS,
            'N_HEAD': NetParameters.N_HEAD, 
            'SEED': SetupParameters.SEED, 'USE_GPU_LOCAL': SetupParameters.USE_GPU_SAMPLE,
            'USE_GPU_GLOBAL': SetupParameters.USE_GPU_TRAIN,
            'NUM_GPU': SetupParameters.NUM_GPU}
