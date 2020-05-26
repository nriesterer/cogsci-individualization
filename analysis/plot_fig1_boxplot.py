""" Produces Figure 1, the boxplot representation of coverage accuracies.

"""

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Parse command-line arguments
if len(sys.argv) != 3:
    print('usage: python plot_fig1_boxplot.py <ccobra-group-csv> <ccobra-indiv-csv>')
    sys.exit(99)

group_csv = sys.argv[1]
indiv_csv = sys.argv[2]

# Read coverage data
group_df = pd.read_csv(group_csv).groupby(['model', 'id'], as_index=False)['hit'].agg('mean')
indiv_df = pd.read_csv(indiv_csv).groupby(['model', 'id'], as_index=False)['hit'].agg('mean')
result_df = pd.concat([group_df, indiv_df])

# Print the median coverage results
print('Median:')
print(result_df.groupby('model', as_index=False)['hit'].agg('median'))

# Load MFA
mfa_median = result_df.loc[result_df['model'] == 'MFA']['hit'].median()
result_df = result_df.loc[result_df['model'] != 'MFA']

# Initialize the plot
sns.set(style='whitegrid', palette='colorblind')
plt.figure(figsize=(7, 3))

col_map = {
    'PHM-Indiv': 'C2',
    'mReasoner-Indiv': 'C0',
    'mReasoner-Group': 'C0',
    'PHM-Group': 'C2'
}

# Plot the coverage data
order = result_df.groupby('model', as_index=False)['hit'].agg('median').sort_values('hit')['model']
color = [col_map[x] for x in order]
sns.boxplot(x='model', y='hit', data=result_df, order=order, palette=color)

# Insert horizontal MFA line
mfa_col = 'C7'
plt.axhline(y=mfa_median, ls='--', color=mfa_col, zorder=10)
plt.text(0.002, mfa_median + 0.015, 'MFA', color=mfa_col, fontsize=10, transform=plt.gca().transAxes)

# Axes definition
plt.xlabel('')
plt.ylabel('Coverage Accuracy')
plt.yticks(np.arange(0, 1.1, 0.1))

# Save and display the plot
plt.tight_layout()
plt.savefig('visualizations/fig1_boxplot.pdf')
plt.show()
