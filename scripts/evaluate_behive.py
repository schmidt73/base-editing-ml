import argparse
import io
import pandas as pd
import numpy as np
import contextlib
import sys
import re
import warnings

#sys.path.append('/home/schmidt73/Dropbox/base-editing/be_hive')
sys.path.append('/lila/data/leslie/schmidth/projects/be_hive')

from be_predict_efficiency import predict as be_efficiency_model
from be_predict_bystander import predict as be_bystander_model

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    yield
    sys.stdout = save_stdout

BASE_EDITORS = [
    'ABE', 'ABE-CP1040', 'BE4', 'BE4-CP1028',
    'AID', 'CDA', 'eA3A', 'evoAPOBEC', 'eA3A-T44DS45A',
    'BE4-H47ES48A', 'eA3A-T31A', 'eA3A-T31AT44A', 'BE4-H47ES48A'
]

CELL_TYPES = ['mES', 'HEK293']

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run the BE-Hive model against a validation set'
    )

    parser.add_argument(
        'validation_set',
        help='CSV file containing base editing outcomes.'
    )
    
    parser.add_argument(
        '-b', '--base-editor',
        choices=BASE_EDITORS, default='BE4'
    )

    parser.add_argument(
        '-c', '--cell-type',
        choices=CELL_TYPES, default='mES'
    )

    parser.add_argument(
        '-o', '--output',
        default=sys.stdout
    )

    parser.add_argument(
        '--num-samples', type=int,
        help='Number of holdout sgRNAs to sample for evaluation'
    )

    return parser.parse_args()

'''
Converts a column name of the form (A|T|C|G)(Position) to
a tuple (Letter, Position).
'''
def parse_column_name(colname):
    match = re.fullmatch(r'(?P<nuc>A|T|C|G)(?P<position>-?\d+)', colname)
    if match is None:
        return None
    return match.group('nuc'), int(match.group('position'))

'''
Parses the outcome into a list of columns and their values
'''
def parse_outcome(sgrna, native_outcome):
    outcomes = {}
    for i in range(len(sgrna)):
        j = i - 19
        if sgrna[i] in 'CG' and j >= -11 and j <= 8:
            outcomes[f'{sgrna[i]}{j}'] = native_outcome[i]
    return outcomes

def be_hive_predict(test_df, mean=0, sigma=1):
    sgrna = test_df.iloc[0]['native_outcome']

    with nostdout():
        efficiency = be_efficiency_model.predict(sgrna)
        pred_df, _ = be_bystander_model.predict(sgrna)

    column_map = {}
    for column in pred_df.columns:
        r = parse_column_name(column)
        if r is not None:
            column_map[column] = r

    def convert_prediction(row):
        result = sgrna
        for column, (_, position) in column_map.items():
            result = result[:position + 19] + row[column] + result[position + 20:]
        return result

    pred_df['outcome'] = pred_df.apply(convert_prediction, axis=1)
    pred_df = pred_df[['outcome', 'Predicted frequency']]

    frequencies = pred_df.merge(test_df, on='outcome')[[
        'sgrna', 'outcome_sgrna', 'native_outcome', 
        'outcome', 'Predicted frequency', 'frequency'
    ]]

    frequencies = frequencies.rename(columns={'Predicted frequency': 'predicted frequency'})

    efficiency = sigmoid(efficiency['Predicted logit score'] * sigma + mean)
    frequencies['predicted frequency'] = frequencies['predicted frequency'] * efficiency

    true_frequency = test_df[test_df.outcome == sgrna].iloc[0]
    true_frequency['predicted frequency'] = 1 - efficiency
    frequencies = frequencies.append(
        true_frequency[frequencies.columns],
        ignore_index=True
    )

    return frequencies

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_arguments()

    validation_df = pd.read_csv(args.validation_set)
    validation_df['outcome'] = validation_df['outcome'].str[1:-5]
    validation_df['native_outcome'] = validation_df['native_outcome'].str[1:-5]

    validation_df['count']     = validation_df[['count_r1', 'count_r2']].sum(axis=1)
    validation_df['total']     = validation_df[['total_r1', 'total_r2']].sum(axis=1)
    validation_df['frequency'] = validation_df['count'] / validation_df['total']

    counts = validation_df[validation_df.native_outcome != validation_df.outcome]\
        [['count_r1', 'count_r2']].sum(axis=1)
    total = validation_df[validation_df.native_outcome != validation_df.outcome]\
        [['total_r1', 'total_r2']].sum(axis=1)

    freq = (counts / total).agg(['mean', 'std'])
    if np.isnan(freq['std']):
        freq['std'] = 1

    with nostdout():
        be_efficiency_model.init_model(base_editor=args.base_editor, celltype=args.cell_type)
        be_bystander_model.init_model(base_editor=args.base_editor, celltype=args.cell_type)

    if args.num_samples is None:
        num_samples = len(validation_df.sgrna_id.drop_duplicates())
    else:
        num_samples = args.num_samples

    sgrnas = set(pd.Series(validation_df.sgrna_id.unique()).sample(num_samples))

    pred_df = validation_df[validation_df.sgrna_id.isin(sgrnas)]
    pred_df = pred_df.groupby('sgrna_id').apply(
        lambda df: be_hive_predict(df, mean=freq['mean'], sigma=freq['std'])
    )
    pred_df.to_csv(args.output)
