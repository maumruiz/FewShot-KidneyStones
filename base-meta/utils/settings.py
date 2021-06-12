import os.path as osp
import pandas as pd

icn = None
way = 5
shot = 5
query = 15
accuracies = []

def save_accs(path):
    test_df = pd.DataFrame()
    test_df['batch'] = [e+1 for e in range(len(accuracies))]
    test_df['acc'] = accuracies
    test_df.to_csv(osp.join(path, 'test_accuracies.csv'), index=False)