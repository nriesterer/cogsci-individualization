import sys

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

# Initialize plotting
sns.set(style='whitegrid', palette='colorblind')

# Plot mreasoner parameter distribution
fig, axs = plt.subplots(2, 2, figsize=(9, 4.5))

bins01 = np.arange(0, 1.2, 0.1) - 0.05
space08 = (8 - 0.1) / 11
bins08 = np.array(list(np.linspace(0.1, 8, 11)) + [8 + space08]) - (0.5 * space08)

#sns.distplot(mreasoner_df['epsilon'], bins=bins01, kde=False, color='C0', ax=axs[0,0])
sns.distplot(mreasoner_df['epsilon'], hist=True, bins=bins01, color='C0', ax=axs[0,0])
axs[0,0].set_title(r'Parameter $\epsilon$')
axs[0,0].set_xlabel('')
axs[0,0].set_ylabel('Density')
axs[0,0].set_xticks(np.linspace(0, 1, 11))
axs[0,0].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, 1, 11)], rotation=90)
# axs[0,0].set_xlim(0, 1)

#sns.distplot(mreasoner_df['lambda'], bins=bins08, kde=False, color='C1', ax=axs[0,1])
sns.distplot(mreasoner_df['lambda'], hist=True, bins=bins08, color='C1', ax=axs[0,1])
axs[0,1].set_title(r'Parameter $\lambda$')
axs[0,1].set_xlabel('')
axs[0,1].set_ylabel('')
axs[0,1].set_xticks(np.linspace(0.1, 8, 11))
axs[0,1].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0.1, 8, 11)], rotation=90)

#sns.distplot(mreasoner_df['omega'], bins=bins01, kde=False, color='C2', ax=axs[1,0])
sns.distplot(mreasoner_df['omega'], hist=True, bins=bins01, color='C2', ax=axs[1,0])
axs[1,0].set_title(r'Parameter $\omega$')
axs[1,0].set_xlabel('')
axs[1,0].set_ylabel('Density')
axs[1,0].set_xticks(np.linspace(0, 1, 11))
axs[1,0].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, 1, 11)], rotation=90)

#sns.distplot(mreasoner_df['sigma'], bins=bins01, kde=False, color='C3', ax=axs[1,1])
sns.distplot(mreasoner_df['sigma'], hist=True, bins=bins01, color='C3', ax=axs[1,1])
axs[1,1].set_title(r'Parameter $\sigma$')
axs[1,1].set_xlabel('')
axs[1,1].set_ylabel('')
axs[1,1].set_xticks(np.linspace(0, 1, 11))
axs[1,1].set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, 1, 11)], rotation=90)

plt.tight_layout()
plt.savefig('visualizations/mreasoner_params.pdf')
plt.show()

# Annotate the data frame with counts of subj param configurations
conf_cnt = mreasoner_df.groupby('id', as_index=False)['score'].agg('count').rename(columns={'score': 'cnt'})
mreasoner_df = pd.merge(mreasoner_df, conf_cnt, on='id')

# Prepare the barplotting data
# barplot_data = []
# for _, series in mreasoner_df.iterrows():
#     for param in ['epsilon', 'lambda', 'omega', 'sigma']:
#         barplot_data.append({
#             'param': param,
#             'value': np.round(series[param], 2),
#             'weight': 1 / series['cnt']
#         })

# barplot_df = pd.DataFrame(barplot_data)

# # Plot the barplot
# fig, axs = plt.subplots(2, 2, figsize=(10, 4))

# sns.barplot(
#     x='value', y='weight', data=barplot_df.loc[barplot_df['param'] == 'epsilon'],
#     estimator=np.sum, color='C0', ax=axs[0, 0])
# axs[0, 0].set_xlabel('')
# axs[0, 0].set_ylabel('Parameter Weight')
# axs[0, 0].set_ylim(0, 15)

# sns.barplot(
#     x='value', y='weight', data=barplot_df.loc[barplot_df['param'] == 'lambda'],
#     estimator=np.sum, color='C1', ax=axs[0, 1])
# axs[0, 1].set_xlabel('')
# axs[0, 1].set_ylabel('')
# axs[0, 1].set_ylim(0, 15)

# sns.barplot(
#     x='value', y='weight', data=barplot_df.loc[barplot_df['param'] == 'omega'],
#     estimator=np.sum, color='C2', ax=axs[1, 0])
# axs[1, 0].set_xlabel('')
# axs[1, 0].set_ylabel('Parameter Weight')
# axs[1, 0].set_ylim(0, 15)

# sns.barplot(
#     x='value', y='weight', data=barplot_df.loc[barplot_df['param'] == 'sigma'],
#     estimator=np.sum, color='C3', ax=axs[1, 1])
# axs[1, 1].set_xlabel('')
# axs[1, 1].set_ylabel('')
# axs[1, 1].set_ylim(0, 15)

# plt.tight_layout()
# plt.show()

var_data = []
for subj_id, subj_df in mreasoner_df.groupby('id'):
    n_epsilon = len(subj_df['epsilon'].unique())
    n_lambda = len(subj_df['lambda'].unique())
    n_omega = len(subj_df['omega'].unique())
    n_sigma = len(subj_df['sigma'].unique())

    ns = {'epsilon': n_epsilon, 'lambda': n_lambda, 'omega': n_omega, 'sigma': n_sigma}

    for param, n in ns.items():
        var_data.append({
            'id': subj_id,
            'param': param,
            'n': n
        })

var_df = pd.DataFrame(var_data)

print()
print('Number of different values per participant per parameter:')
print(var_df.groupby('param', as_index=False)['n'].agg('mean'))
print()

# sns.barplot(x='param', y='n', data=var_df)
# plt.show()
