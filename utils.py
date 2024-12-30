import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

def plot_two_var(df, common_var, var1, var2):
    plt.figure(figsize=(9,4))
    sns.barplot(df, x="common_var", y="var1", color="blue", alpha=0.6, label="magnitude counts")
    sns.barplot(df, x="common_var", y="var2", color="yellow", alpha=0.5, label="tsunami counts")

    # plt.title("Comparison of Total Values and Tsunami Counts by Magnitude")
    # plt.xlabel("Magnitude")
    # plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def impute_na(data, variable):
    df = data.copy()
    df[variable+'_random'] = df[variable]
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
    return df[variable+'_random']

def diagnostic_plots(df, variable):
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)

    plt.show()    

def log_transformation(df, var):
    df[var] = np.log(df[var] + 1)

def reciprocal_transformatio(df, var):
    df[var] = 1/(df[var]+1)

def sq_root_transformation(df, var):
    df[var] = np.sqrt(df[var])

def exp_transformation(df, var):
    df[var] = df[var]**(1/5)

def boxcox_transformation(df, var):
    df[var], params = stats.boxcox(df[var] + 1)
    print("optimal lambda; ", params)
