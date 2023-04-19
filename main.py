from env import *
from dqn import *

from settings import *
from params import *
from supply import *
from demand import *

# NOTES
# We currently assume that each requested unit is a separate request.
# Compatibility now only on major antigens -> specify to include patient group specific mandatory combinations.


def main():

    SETTINGS = Settings()
    PARAMS = Params(SETTINGS)
    SETTINGS.minor = PARAMS.minor

    paths = [
        "results", f"results/{SETTINGS.model_name}_{''.join(PARAMS.minor)}", f"results/{SETTINGS.model_name}_{''.join(PARAMS.minor)}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}", 
        "models", f"models/{SETTINGS.model_name}_{''.join(PARAMS.minor)}", f"models/{SETTINGS.model_name}_{''.join(PARAMS.minor)}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}"]
    for path in paths:
        SETTINGS.check_dir_existence(SETTINGS.home_dir + path)

    print("CREATING ENVIRONMENT")
    env = MatchingEnv(SETTINGS, PARAMS)
    print("CREATING DQN")
    dqn = DQN(SETTINGS, env)

    print(f"model: {SETTINGS.model_name}, minor antigens:{', '.join(PARAMS.minor)}")
    print(f"alpha: {SETTINGS.alpha}, gamma: {SETTINGS.gamma}, batch size: {SETTINGS.batch_size}.")

    # Train the agent
    if SETTINGS.RL_mode == "train":
        dqn.train(SETTINGS, PARAMS)
    elif SETTINGS.RL_mode == "train_minrar":
        dqn.train_minrar(SETTINGS, PARAMS)
    elif SETTINGS.RL_mode == "test":
        dqn.test(SETTINGS, PARAMS)
    else:
        print(f"Mode '{SETTINGS.RL_mode}' is not implemented. Please check this parameter in 'settings.py'.")



if __name__ == "__main__":
    main()