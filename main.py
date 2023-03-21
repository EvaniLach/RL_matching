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

    paths = [
        "results", f"results/{SETTINGS.model_name}", f"results/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}", 
        "models", f"models/{SETTINGS.model_name}", f"models/{SETTINGS.model_name}/a{SETTINGS.alpha}_g{SETTINGS.gamma}_b{SETTINGS.batch_size}"]
    for path in paths:
        SETTINGS.check_dir_existence(SETTINGS.home_dir + path)


    # Sample demand for each day in the simulation, and write to a csv file.
    if SETTINGS.mode == "demand":
        for htype in SETTINGS.n_hospitals.keys():
            for _ in range(SETTINGS.n_hospitals[htype]):
                generate_demand(SETTINGS, PARAMS, htype, SETTINGS.avg_daily_demand[htype])

    # Sample RBC units to be used as supply in the simulation, and write to csv file.
    elif SETTINGS.mode == "supply":
        generate_supply(SETTINGS, PARAMS)

    # Run the simulation, using either linear programming or reinforcement learning to determine the matching strategy.
    elif SETTINGS.mode == "train":

        print("CREATING ENVIRONMENT")
        env = MatchingEnv(SETTINGS, PARAMS)
        print("CREATING DQN")
        dqn = DQN(SETTINGS, env)

        print(f"alpha: {SETTINGS.alpha}, gamma: {SETTINGS.gamma}, batch size: {SETTINGS.batch_size}.")

        # Train the agent
        dqn.train(SETTINGS, PARAMS)



if __name__ == "__main__":
    main()