import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if len(sys.argv) != 2:
    print('usage: python visualize_boxplot.py <ccobra-result-csv>')
    sys.exit(99)

ccobra_csv = sys.argv[1]

# Load prediction data
result_df = pd.read_csv(ccobra_csv).groupby(['model', 'id'], as_index=False)['hit'].agg('mean')
result_df = result_df.replace({
    'PyPHM': 'PHM-Fit',
    'PHM': 'PHM',
    'mReasoner': 'mReasoner-Indiv',
    'mReasoner\'': 'mReasoner'
})
result_df = result_df.loc[result_df['model'] == 'mReasoner-Indiv']

# Load pretrain data
res_pretr_df = pd.read_csv('premodels.csv').groupby(['model', 'id'], as_index=False)['hit'].agg('mean')
res_pretr_df = res_pretr_df.replace({
    'PHM-Person': 'PHM-Indiv',
    'PHM-Pre': 'PHM-Group',
    'mReasoner-Pretrain': 'mReasoner-Group'
})

# Load additional mReas data
mres_df = pd.read_csv('2020-02-11-mReasoner-group.csv').groupby(['model', 'id'], as_index=False)['hit'].agg('mean')
mres_df = mres_df.replace({
    'mReasoner': 'mReasoner-Group2'
})

result_df = pd.concat([result_df, res_pretr_df, mres_df])

print('Median:')
print(result_df.groupby('model', as_index=False)['hit'].agg('median'))

# Load MFA
mfa_median = result_df.loc[result_df['model'] == 'MFA']['hit'].median()
result_df = result_df.loc[result_df['model'] != 'MFA']

# Plot stuff
sns.set(style='whitegrid', palette='colorblind')
plt.figure(figsize=(7, 3))

col_map = {
    'PHM-Indiv': 'C2',
    'mReasoner-Indiv': 'C0',
    'mReasoner-Group': 'C0',
    'PHM-Group': 'C2',
    'mReasoner-Group2': 'C3'
}

order = result_df.groupby('model', as_index=False)['hit'].agg('median').sort_values('hit')['model']
color = [col_map[x] for x in order]
sns.boxplot(x='model', y='hit', data=result_df, order=order, palette=color)

mfa_col = 'C7'
plt.axhline(y=mfa_median, ls='--', color=mfa_col, zorder=10)
plt.text(0.002, mfa_median + 0.015, 'MFA', color=mfa_col, fontsize=10, transform=plt.gca().transAxes)

plt.xlabel('')
plt.ylabel('Coverage Accuracy')
plt.yticks(np.arange(0, 1.1, 0.1))

plt.tight_layout()
plt.savefig('visualizations/accuracy.pdf')
plt.show()