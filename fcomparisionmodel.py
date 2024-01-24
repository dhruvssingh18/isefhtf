import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load datasets (replace 'dataset1.csv' and 'dataset2.csv' with your file paths)
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')

# Basic Descriptive Statistics
desc_stats_1 = df1.describe()
desc_stats_2 = df2.describe()

# Visualization - Histograms
for column in df1.columns:
    plt.figure(figsize=(10, 4))
    plt.hist(df1[column], alpha=0.5, label='Dataset 1', bins=20)
    plt.hist(df2[column], alpha=0.5, label='Dataset 2', bins=20)
    plt.title(f'Histogram of {column}')
    plt.legend()
    plt.show()

# T-test for each feature
t_test_results = {}
for column in df1.columns:
    t_stat, p_value = ttest_ind(df1[column].dropna(), df2[column].dropna())
    t_test_results[column] = {'t_statistic': t_stat, 'p_value': p_value}

# Display Results
print("Descriptive Statistics - Dataset 1:")
print(desc_stats_1)

print("\nDescriptive Statistics - Dataset 2:")
print(desc_stats_2)

print("\nT-test Results:")
for column, result in t_test_results.items():
    print(f"{column}: t-statistic = {result['t_statistic']}, p-value = {result['p_value']}")
