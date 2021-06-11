import argparse
import os
import re
import pandas as pd
import random
import math

default_split = {
    'training':   80,
    'test':       10,
    'validation': 10
}

class SplitAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        m = re.fullmatch(r'(\d+):(\d+):(\d+)', values) 

        if m is None:
            raise argparse.ArgumentTypeError('Format must be of integral TRAINING:TEST:VALIDATION')

        training, test, validation = list(map(int, m.groups()))
        if training + test + validation != 100:
            raise argparse.ArgumentTypeError('Split size must sum to 100')

        setattr(args, self.dest, {
            'training': training,
            'test': test,
            'validation': validation
        })

def parse_args():
    p = argparse.ArgumentParser(
        "Splits data into training/test/validation set"
    )

    p.add_argument(
        '-s', '--split',
        help='Training/test/validation split size',
        default=default_split, action=SplitAction
    )

    p.add_argument(
        '-r', '--random',
        help='Randomness seed to use for splitting',
        type=int
    )

    p.add_argument(
        'dataset',
        help='Full dataset CSV to split'
    )

    p.add_argument(
        '-p', '--prefix',
        help='Output prefix to use for split datasets'
    )

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.dataset)
    sgrnas = list(df['sgrna_id'].drop_duplicates())

    if args.random is not None:
        random.seed(args.random)

    for key in args.split.keys():
        args.split[key] = math.ceil(len(sgrnas) * (args.split[key] / 100))

    random.shuffle(sgrnas) 
    training_sgrnas = set(sgrnas[:args.split['training']])
    test_sgrnas = set(sgrnas[args.split['training']:args.split['training'] + args.split['test']])
    validation_sgrnas = set(sgrnas[args.split['training'] + args.split['test']:])

    training_df = df[df.sgrna_id.isin(training_sgrnas)]
    test_df = df[df.sgrna_id.isin(test_sgrnas)]
    validation_df = df[df.sgrna_id.isin(validation_sgrnas)]

    if args.prefix is not None:
        prefix = args.prefix
    else:
        prefix = os.path.splitext(args.dataset)[0]

    training_df.to_csv(prefix + '_training.csv')
    test_df.to_csv(prefix + '_test.csv')
    validation_df.to_csv(prefix + '_validation.csv')
