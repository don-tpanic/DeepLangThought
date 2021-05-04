import argparse
from TRAIN import train_language_model
from TRAIN import train_language_modelWithSuperGroups
from TRAIN import train


parser = argparse.ArgumentParser()
parser.add_argument('--option', dest='run')
args = parser.parse_args()


if __name__ == '__main__':
    if args.run == 'finegrain':
        train_language_model.execute()
    elif args.run == 'coarsegrain':
        train_language_modelWithSuperGroups.execute()

    # TODO: test 
    elif args.run == 'test':
        train.execute()