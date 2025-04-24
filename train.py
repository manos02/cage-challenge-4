from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFixedActionWrapper
from CybORG.Agents.Wrappers import BlueFlatWrapper
from Trainers.ippo_RLIB import CC4EnvManager
import argparse
import warnings


# code from https://github.com/Davide236/Collaborative_RL_CAGE_4/blob/main/train.py
# File made to train different RL algorithms on the CybORG environment
def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Process input parameters for agent training")

    # Add arguments
    parser.add_argument(
        '--Method',
        type=str,
        default='IPPO',
        help='The method to be used (default: IPPO)'
    )
    
    parser.add_argument(
        '--Messages',
        type=bool,
        default=False,
        help='Boolean flag for messages (default: False)'
    )

    parser.add_argument(
        '--Load_last',
        type=bool,
        default=False,
        help='Boolean flag for loading the last saved network (default: False)'
    )

    parser.add_argument(
        '--Load_best',
        type=bool,
        default=False,
        help='Boolean flag for loading the best saved network (default: False)'
    )

    parser.add_argument(
        '--Rollout',
        type=int,
        default=10,
        help='Integer Number to determine the number of episodes stored before training (default: 10)'
    )

    parser.add_argument(
        '--Episodes',
        type=int,
        default=4000,
        help='Integer Number to determine the total number of training episodes (default: 4000)'
    )
    # Parse the arguments
    args = parser.parse_args()
    method = args.Method
    print(f'Using method: {method}, loading network: {args.Load_best | args.Load_last}, Using messages: {args.Messages}')
    

    # trainer = PPOTrainer(args)
    # trainer.run()

    trainer = CC4EnvManager()
    trainer.run()



if __name__ == "__main__":
    main()