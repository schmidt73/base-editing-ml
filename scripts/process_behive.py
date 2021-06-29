import pandas as pd
import numpy as np
import argparse
import pickle
import re
import math
import sys

from funcy import take

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=('Processes raw BE-Hive data for a particular screen to'
                     'a unified Pandas DF.')
    )

    parser.add_argument(
        'bystander', type=argparse.FileType('rb'),
        help='A pickle file containing the bystander data')

    parser.add_argument(
        'efficiency', type=argparse.FileType('r'),
        help='A CSV file containing the editing efficiency data'
    )

    parser.add_argument(
        '-o', '--outfile', type=argparse.FileType('w'),
        default=sys.stdout, help='Name of output CSV'
    )

    return parser.parse_args()

def get_replicate_names(efficiency_df):
    rep1_total_col, rep2_total_col = list(filter(lambda x: x.startswith('Total count'), efficiency_df.columns))
    n = len('Total count ')
    return (rep1_total_col[n:], rep2_total_col[n:])

def parse_row(bystander_map, replicate_names, row):
    rep1, rep2 = replicate_names

    sgrna_id = row['Name (unique)']
    sgrna = row['gRNA (20nt)']
    context = row['Sequence context (56nt)']

    row = row.fillna(0)

    rep1_total_edited, rep2_total_edited = list(filter(lambda x: x.startswith('Edited count'), row.keys()))

    rep1_total, rep2_total = row[f'Total count {rep1}'], row[f'Total count {rep2}']
    rep1_edited, rep2_edited = row[f'Edited count {rep1}'], row[f'Edited count {rep2}']

    if sgrna_id not in bystander_map:
        return None

    bystander_df = bystander_map[sgrna_id]
    
    if bystander_df.columns.isin([f"Count_{rep1}", f"Count_{rep2}"]).sum() != 2:
        return None

    bystander_edit_regex = r'(A|T|C|G)(-?)[0-9]+'
    positions = filter(lambda s: re.match(bystander_edit_regex, s),
                        bystander_map[sgrna_id])

    positions = [(pos, int(re.findall('-?[0-9]+', pos)[0]) + 21) for pos in positions]

    # 0-based pos 21 in context string = 1-based pos 1 in bystander_df
    def convert_bystander_to_outcome(bs_row):
        edited_outcome = context
        for name, pos in positions:
            edited_outcome = f"{edited_outcome[:pos]}{bs_row[name]}{edited_outcome[pos+1:]}"
        return edited_outcome

    bystander_df = bystander_df.rename(columns={
        f"Count_{rep1}": "count_r1",
        f"Count_{rep2}": "count_r2",
    })

    bystander_df['outcome'] = bystander_df.apply(convert_bystander_to_outcome, axis=1)

    # Filter out rare outcomes 
    bystander_df = bystander_df[bystander_df.count_r1 + bystander_df.count_r2 >= 100]

    total_edited_r1 = np.sum(bystander_df['count_r1'])
    total_edited_r2 = np.sum(bystander_df['count_r2'])

    native_outcome = {
        'outcome': context,
        'count_r1': rep1_total - total_edited_r1,
        'count_r2': rep2_total - total_edited_r2,
    }

    bystander_df = bystander_df.append(native_outcome, ignore_index=True).reset_index()

    bystander_df['outcome_sgrna'] = bystander_df['outcome']
    bystander_df['outcome'] = bystander_df['outcome']

    bystander_df['total_edited_r1'] = total_edited_r1
    bystander_df['total_edited_r2'] = total_edited_r2
    bystander_df['total_r1'] = rep1_total
    bystander_df['total_r2'] = rep2_total

    '''
    Assertions to verify that PKL and CSV are synchronized, have had
    issues with CSV files being corrupted.
    '''
    assert bystander_df['total_edited_r1'].iloc[0] <= rep1_edited, f"Failed rep1 total assertion for: {sgrna_id}"
    assert bystander_df['total_edited_r2'].iloc[0] <= rep2_edited, f"Failed rep2 total assertion for: {sgrna_id}"

    bystander_df['native_outcome'] = context
    bystander_df['sgrna'] = sgrna
    bystander_df['sgrna_id'] = sgrna_id

    return bystander_df[[
        'sgrna_id', 'sgrna', 'native_outcome', 'outcome_sgrna', 
        'outcome', 'total_r1', 'total_r2', 'count_r1', 'count_r2'
    ]]
    
def main():
    args = parse_arguments()

    efficiency_df = pd.read_csv(args.efficiency)
    bystander_map = pickle.load(args.bystander)

    replicate_names = get_replicate_names(efficiency_df)
    process_row = lambda row: parse_row(bystander_map, replicate_names, row)

    bystander_map['satmut_6mer_7'].to_csv('../test.csv')

    dfs = efficiency_df.apply(process_row, axis=1)
    df = pd.concat(list(dfs))
    df.to_csv(args.outfile)

if __name__ == "__main__":
    main()
