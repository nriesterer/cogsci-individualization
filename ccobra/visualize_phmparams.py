import sys
import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Parse command line arguments
if len(sys.argv) != 2:
    print('usage: python3 visualize.py <output-file>')
    sys.exit(99)

output_filename = sys.argv[1]

# Load the output content
mreasoner_data = []
phm_data = []
with open(output_filename) as out_file:
    modelname = None
    mreas_params = None
    for line in out_file.readlines():
        line = line.strip()

        # Identify current model
        if line.startswith('Evaluating '):
            print(line)
            if 'ccobra_mreasoner.py' in line:
                modelname = 'mReasoner'
            elif 'phm' in line:
                modelname = 'PHM'
            # TODO: Add other models
            continue

        # Fit PHM output
        line = line.replace('p_entailm', '\'p_entailm\'').replace('direction', '\'direction\'').replace('max_confi', '\'max_confi\'')

        line = line.split()

        if line[0] == 'Fit':
            # Read the fit result line
            content = [x.split('=') for x in line if '=' in x]
            content = dict([(x, eval(y)) for x, y in content])
            content['params'] = dict(content['params'])

            # Populate result data
            if modelname == 'mReasoner':
                mreas_params = content['params']

                mreasoner_data.append({
                    'id': content['id'],
                    'model': modelname,
                    'score': content['score'],
                    'epsilon': content['params']['epsilon'],
                    'lambda': content['params']['lambda'],
                    'omega': content['params']['omega'],
                    'sigma': content['params']['sigma']
                })
            elif modelname == 'PHM':
                content['params']['max_confi'] = dict(content['params']['max_confi'])

                phm_data.append({
                    'id': content['id'],
                    'model': modelname,
                    'p_ent': content['params']['p_entailm'],
                    'conf_A': content['params']['max_confi']['A'],
                    'conf_I': content['params']['max_confi']['I'],
                    'conf_E': content['params']['max_confi']['E'],
                    'conf_O': content['params']['max_confi']['O']
                })
        elif line[0].startswith('[('):
            assert len(line) == 1 and modelname == 'mReasoner'
            content = dict(eval(line[0]))

            # Omit the duplicate parameterization
            if content == mreas_params:
                continue

            # Read the alternative parameter line
            mreasoner_data.append({
                'id': mreasoner_data[-1]['id'],
                'model': modelname,
                'score': mreasoner_data[-1]['score'],
                'epsilon': content['epsilon'],
                'lambda': content['lambda'],
                'omega': content['omega'],
                'sigma': content['sigma']
            })

phm_df = pd.DataFrame(phm_data)
mreasoner_df = pd.DataFrame(mreasoner_data)[[
    'id', 'model', 'score', 'epsilon', 'lambda', 'omega', 'sigma']]

# Compute
dat = []
cols = ['conf_A', 'conf_E', 'conf_I', 'conf_O', 'p_ent']
for col in cols:
    cnt = dict(collections.Counter(phm_df[col]))
    dat.append({
        'col': col,
        '0_val': -cnt[0] / 139,
        '1_val': cnt[1] / 139
    })
df = pd.DataFrame(dat)

sns.set(style='whitegrid', palette='colorblind')

plt.figure(figsize=(7, 3))
sns.barplot(
    y='col', x='0_val', data=df, orient='h', color='C0',
    order=['conf_A', 'conf_I', 'conf_E', 'conf_O', 'p_ent'])
sns.barplot(
    y='col', x='1_val', data=df, orient='h', color='C1',
    order=['conf_A', 'conf_I', 'conf_E', 'conf_O', 'p_ent'])

target_x = np.array([-139] + list(np.arange(-120, 121, 20)) + [139])
source_x = target_x / 139
plt.xticks(source_x, [str(np.abs(np.round(x, 2))) for x in target_x])
plt.xlim(-1, 1)

plt.ylabel('')
plt.xlabel('Number of Reasoners')

plt.yticks([0, 1, 2, 3, 4], cols, style='italic')

from matplotlib.lines import Line2D

legend_handle = [
    Line2D([0], [0], color='w', marker='o', markerfacecolor='C0', ms=12, label=r'$param=0$'),
    Line2D([0], [0], color='w', marker='o', markerfacecolor='C1', ms=12, label=r'$param=1$'),
]
plt.legend(
    handles=legend_handle, bbox_to_anchor=(0, 1.1, 1, 0.05), loc='upper center',
    ncol=2, borderaxespad=0, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 1.05])
plt.savefig('visualizations/phmparams-horizontal.pdf')
plt.show()

