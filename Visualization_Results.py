import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Root directory where experiments are stored
root_dir = 'experimental_results'

# Search pattern for all numerical_results.csv files
search_pattern = os.path.join(root_dir, 'experiment_*', '*', '*', 'numerical_results.csv')
csv_files = glob.glob(search_pattern)
structured_results = defaultdict(dict)

# Columns of Pandas Dataframe df
columns = ['auc_exp', 'auc_low', 'max_exp', 'max_low', 'maximal_reward', 'MAE', 'proportion_policy_correct', 'experiment', 'grid', 'agent']
df = pd.DataFrame(columns=columns)

# Summarize all data into one dataframe df.
for path in csv_files:
    parts = path.split(os.sep)
    experiment = parts[1]          
    if experiment == "experiment_deprecated":
        continue
    grid = parts[2]                
    agent = parts[3] 
    numerical_results = list(pd.read_csv(path).iloc[0, :])
    numerical_results.extend([experiment, grid, agent])
    df.loc[len(df)] = numerical_results

# Change the names of the values in some attributes
df['grid'] = df['grid'].replace({'grid_A1': 'A1', 'grid_custom': 'Custom'})
df['experiment'] = df['experiment'].replace({'experiment_1': 'exp1','experiment_2': 'exp2','experiment_3': 'exp3','experiment_4': 'exp4','experiment_5': 'exp5','experiment_6': 'exp6'})
df['agent'] = df['agent'].replace({'mc_control': 'MC', 'q_learning': 'QL'})
print(df.head())

# Add labels for heatmap
df['Grid-Agent'] = df['grid'] + '-' + df['agent']
df['Experiment'] = df['experiment']

# Define metrics and titles
metrics = ['auc_exp', 'auc_low', 'MAE', 'proportion_policy_correct']
titles = [r'$AUC_{EXP}$', r'$AUC_{LCB}$', r'$MAE$', r'$PPC$']


# Make the Heatmap Visualization based on data in df. 
fig, axs = plt.subplots(2,2, figsize=(16, 12))

for i, metric in enumerate(metrics):
    row = i // 2
    col = i % 2

    pivot_table = df.pivot(index='Experiment', columns='Grid-Agent', values=metric)
    sns.heatmap(pivot_table, ax=axs[row, col], annot=True, cmap='viridis', fmt=".2f", annot_kws={"size": 16})

    axs[row, col].set_title(titles[i], fontsize=23)
    axs[row, col].set_xlabel('Grid-Agent' if i > 1 else '', fontsize=18)
    axs[row, col].set_ylabel('Experiments' if i in [0, 2] else '', fontsize=18)
    axs[row, col].tick_params(axis='x', labelrotation=45, labelsize=14)
    axs[row, col].tick_params(axis='y', labelrotation=45, labelsize=14)


fig.tight_layout()
fig.savefig('All_Results_in_Heatmap.png')

plt.show()
