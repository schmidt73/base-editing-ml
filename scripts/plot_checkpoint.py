import seaborn as sns
import matplotlib.pyplot as plt    
import pandas as pd
from seaborn.palettes import color_palette
import torch
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Plots the training/test loss")
    p.add_argument(
        'checkpoint',
        help='Pytorch checkpoint file'
    )
    p.add_argument(
        '-o', '--output',
        help='Path of output file, otherwise will display result graphically'
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location=torch.device('cpu')
    )


    loss_df = pd.DataFrame(
        {'Training Loss': checkpoint['training_losses'],
         'Test Loss': checkpoint['test_losses']}
    ).reset_index()

    loss_df = loss_df.melt(
        id_vars=["index"] 
    )

    loss_df = loss_df.rename(columns={
        'index': 'Epoch',
        'variable': 'Variable',
        'value': 'Value'
    })

    sns.set_theme(context='notebook', style='ticks', palette='rocket_r')
    g = sns.relplot(data=loss_df, x='Epoch', y='Value', hue='Variable', kind='line')

    if args.output is None:
       plt.show() 
