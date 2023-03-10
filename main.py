from env import *
from dqn import *

import sys
sys.path.insert(1, 'C:/Users/Merel/Documents/Sanquin/Projects/RBC matching/Paper patient groups/blood_matching/')
from settings import *
from params import *

# NOTES
# We currently assume that each requested unit is a separate request.
# Compatibility now only on major antigens -> specify to include patient group specific mandatory combinations.


def main():

    SETTINGS = Settings()
    PARAMS = Params(SETTINGS)

    for path in ["results", "results/"+SETTINGS.model_name]:
        check_dir_existence(path)

    # Get the hospital's type ('regional' or 'university')
    htype = max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])

    print("CREATING ENVIRONMENT")
    env = MatchingEnv(SETTINGS, PARAMS, htype)
    print("CREATING DQN")
    dqn = DQN(SETTINGS, env)

    # Train the agent
    dqn.train(SETTINGS, PARAMS, htype)



if __name__ == "__main__":
    main()