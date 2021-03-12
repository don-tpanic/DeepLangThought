import argparse
from EVAL import compare_semantic_activations
from EVAL import results_vis


parser = argparse.ArgumentParser()
parser.add_argument('--option', dest='run')
args = parser.parse_args()


if __name__ == '__main__':
    if args.run == 'eval':
        print('***** evaluating intermediate results used for plotting *****')
        compare_semantic_activations.execute()
    elif args.run == 'plot':
        print('***** plotting figures *****')
        results_vis.execute()
