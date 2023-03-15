from env import *
from dqn import *

import sys
# sys.path.insert(1, 'C:/Users/Merel/Documents/Sanquin/Projects/RBC matching/Paper patient groups/blood_matching/')
from settings import *
from params import *

# NOTES
# We currently assume that each requested unit is a separate request.
# Compatibility now only on major antigens -> specify to include patient group specific mandatory combinations.


def main():

    SETTINGS = Settings()
    PARAMS = Params(SETTINGS)

    for path in ["results", "results/"+SETTINGS.model_name]:
        SETTINGS.check_dir_existence(path)

    print("CREATING ENVIRONMENT")
    env = MatchingEnv(SETTINGS, PARAMS)
    print("CREATING DQN")
    dqn = DQN(SETTINGS, env)

    # Train the agent
    dqn.train(SETTINGS, PARAMS)



if __name__ == "__main__":
    main()