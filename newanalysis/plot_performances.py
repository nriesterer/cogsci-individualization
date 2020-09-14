import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if len(sys.argv) != 3:
    print('usage: python plot_performances.py <group_csv> <indiv_csv>')
    exit()

group_file = sys.argv[1]
indiv_file = sys.argv[2]

# Load the data
df_group = pd.read_csv(group_file)
df_indiv = pd.read_csv(indiv_file)
df = pd.concat([df_group, df_indiv], sort=True)

# Prepare the data for plotting
plot_df = df.groupby(['model', 'id'], as_index=False)['hit'].agg('mean')
mfa_df = plot_df.loc[plot_df['model'] == 'MFA']
mfa_median = mfa_df['hit'].median()
plot_df = plot_df.loc[plot_df['model'] != 'MFA']

# Plot the data
sns.set(style='whitegrid', palette='colorblind')
plt.figure(figsize=(7, 3))

order = plot_df.groupby('model', as_index=False)['hit'].agg('median').sort_values('hit')['model']
colors = [('C0' if 'mReasoner' in x else 'C2') for x in order]
sns.boxplot(x='model', y='hit', data=plot_df, order=order, palette=colors)

plt.axhline(y=mfa_median, ls='--', color='C7', zorder=10)
plt.text(0.002, mfa_median + 0.015, 'MFA', color='C7', fontsize=10, transform=plt.gca().transAxes)

plt.xlabel('')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel('Coverage Accuracy')

plt.tight_layout()
plt.savefig('visualizations/performances.pdf')
plt.show()
