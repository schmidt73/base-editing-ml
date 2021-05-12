import time
import pandas as pd
import sys

sys.path.append('scripts')
import model

from funcy import take

df = pd.read_csv('data/be-hive/processed/mES_12kChar_BE4_H47ES48A_test.csv')

def time_batching():
    start = time.time() 
    list(take(5, model.create_batches(df)))
    elapsed = time.time() - start
    return elapsed

print(f'Time taken: {time_batching()}')
