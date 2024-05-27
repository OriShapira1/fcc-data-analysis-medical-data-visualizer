import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)
overweight_mask = df['overweight'] > 25
df.loc[overweight_mask, 'overweight'] = 1 
df.loc[~(overweight_mask), 'overweight'] = 0

# 3
chol_mask = df['cholesterol'] > 1
gluc_mask = df['gluc'] > 1
df.loc[chol_mask, 'cholesterol'] = 1
df.loc[~(chol_mask), 'cholesterol'] = 0
df.loc[gluc_mask, 'gluc'] = 1
df.loc[~(gluc_mask), 'gluc'] = 0


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable']).value_counts()
    
    # 7
    df_cat = df_cat.reset_index()
    df_cat = df_cat.rename(columns={0: 'total'})
    plot = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        kind='bar',
        col='cardio',
        hue='value',
    )


    # 8
    fig = plot.figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # clean by height and weight
    df_heat = df[(df['height'] >= df['height'].quantile(0.025)) &  
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]
    # clean invalid pressures
    df_heat = df_heat[(df_heat['ap_lo'] <= df_heat['ap_hi'])]
    

    # 12
    corr = df_heat.corr()


    # 13
    mask = np.rot90(np.tri(14, dtype=bool), 2)


    # 14
    fig, ax = plt.subplots(figsize=(10, 8))


    # 15
    sns.heatmap(
        data=corr,
        mask=mask,
        ax=ax,
        square=True,
        cbar=True,
        annot=True,
        fmt='.1f',
        linewidths=0.2,
    )
    fig.tight_layout(pad=4)

    # 16
    fig.savefig('heatmap.png')
    return fig
