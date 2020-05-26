import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if len(sys.argv) != 2:
    print('usage: python plot_fig4_congruency.py <ccobra-result-csv>')
    sys.exit(99)

# Load data
result_df = pd.read_csv(sys.argv[1])
result_df = result_df.groupby(['id', 'model'], as_index=False)['hit'].agg('mean')

# Compute order of individuals for each model
order_phm = result_df.loc[result_df['model'] == 'PHM-Indiv'].sort_values('hit')['id'].values
order_mre = result_df.loc[result_df['model'] == 'mReasoner-Indiv'].sort_values('hit')['id'].values

# Compare the quartiles
n_indiv_per_quartile = int(np.ceil(len(order_phm) / 4))
match_data = []
for idx in range(4):
    start = n_indiv_per_quartile * idx
    end = start + n_indiv_per_quartile

    quart_phm = order_phm[start:end]
    quart_mre = order_mre[start:end]
    assert len(quart_phm) == len(quart_mre)

    # Compute amount of overlap
    n_matches = len(set(quart_phm).intersection(set(quart_mre)))
    match_pc = (n_matches / len(quart_mre)) * 100

    match_data.append({
        'quartile': idx + 1,
        'n_matches': n_matches,
        'pc_matches': match_pc
    })

    print('Quartile {}: {} matches ({:.2f}%)'.format(idx + 1, n_matches, match_pc))

match_df = pd.DataFrame(match_data)

sns.set(style='whitegrid', palette='colorblind')
plt.figure(figsize=(7, 3.5))
sns.barplot(x='quartile', y='pc_matches', data=match_df, color='C0')

# Plot text labels
offset = -2.5
for _, series in match_df.iterrows():
    quart_idx = series['quartile'] - 1
    pc = series['pc_matches']
    plt.text(quart_idx, pc + offset, '{:.1f}'.format(np.round(pc, 1)), color='white', ha='center', va='top', fontsize=11)

plt.yticks(np.arange(0, 101, 10))
plt.ylabel('Percentage of Congruency')
plt.xlabel('Model Performance Quartile')

plt.tight_layout()
plt.savefig('visualizations/fig4_congruency.pdf')
plt.show()
