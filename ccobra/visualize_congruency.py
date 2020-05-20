import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
result_df = pd.read_csv('2020-01-23-grid-11-5.csv')
result_df = result_df.groupby(['id', 'model'], as_index=False)['hit'].agg('mean')

# Compute order of individuals for each model
order_phm = result_df.loc[result_df['model'] == 'PyPHM'].sort_values('hit')['id'].values
order_mre = result_df.loc[result_df['model'] == 'mReasoner'].sort_values('hit')['id'].values

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

plt.yticks(np.arange(0, 101, 10))
plt.ylabel('Percentage of Congruency')
plt.xlabel('Model Performance Quartile')

plt.tight_layout()
plt.savefig('visualizations/matches.pdf')
plt.show()
