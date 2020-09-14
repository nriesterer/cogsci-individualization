import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if len(sys.argv) != 2:
    print('usage: python plot_mReasoner_params.py <mReasoner-log>')
    exit()

logfile = sys.argv[1]

# Read the log file
json_dict = None
with open(logfile) as lf:
    json_dict = json.load(lf)

# Convert log to histogram vectors
values = {}
for dude, log in json_dict.items():
    params = log['best_params']
    for param_conf in params:
        for param, value in param_conf.items():
            if param not in values:
                values[param] = []
            values[param].append(value)

# Initialize plotting
sns.set(style='whitegrid', palette='colorblind')
fig, axs = plt.subplots(2, 2, figsize=(9, 4.5))

# Bin definition
bins01 = np.arange(0, 1.2, 0.1) - 0.05
space08 = (8 - 0.1) / 11
bins08 = np.array(list(np.linspace(0.1, 8, 11)) + [8 + space08]) - (0.5 * space08)

# Plot epsilon
sns.distplot(values['epsilon'], hist=True, bins=bins01, color='C0', ax=axs[0,0])
axs[0,0].set_title(r'Parameter $\epsilon$')
axs[0,0].set_xlabel('')
axs[0,0].set_ylabel('Density')
axs[0,0].set_xticks(np.linspace(0, 1, 11))
axs[0,0].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, 1, 11)], rotation=90)

# Plot lambda
sns.distplot(values['lambda'], hist=True, bins=bins08, color='C1', ax=axs[0,1])
axs[0,1].set_title(r'Parameter $\lambda$')
axs[0,1].set_xlabel('')
axs[0,1].set_ylabel('')
axs[0,1].set_xticks(np.linspace(0.1, 8, 11))
axs[0,1].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0.1, 8, 11)], rotation=90)

# Plot omega
sns.distplot(values['omega'], hist=True, bins=bins01, color='C2', ax=axs[1,0])
axs[1,0].set_title(r'Parameter $\omega$')
axs[1,0].set_xlabel('')
axs[1,0].set_ylabel('Density')
axs[1,0].set_xticks(np.linspace(0, 1, 11))
axs[1,0].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, 1, 11)], rotation=90)

# Plot sigma
sns.distplot(values['sigma'], hist=True, bins=bins01, color='C3', ax=axs[1,1])
axs[1,1].set_title(r'Parameter $\sigma$')
axs[1,1].set_xlabel('')
axs[1,1].set_ylabel('')
axs[1,1].set_xticks(np.linspace(0, 1, 11))
axs[1,1].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, 1, 11)], rotation=90)

# Save and display plot
plt.tight_layout()
plt.savefig('visualizations/mReasoner_params.pdf')
plt.show()
