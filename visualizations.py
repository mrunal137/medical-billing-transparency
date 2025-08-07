import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(series, title):
    plt.figure(figsize=(8,5))
    sns.histplot(series, kde=True)
    plt.title(title)
    plt.show()

def plot_bar(data, x, y, title):
    plt.figure(figsize=(10,6))
    sns.barplot(x=x, y=y, data=data, estimator='mean')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def plot_box(data, x, y, title):
    plt.figure(figsize=(12,6))
    sns.boxplot(x=x, y=y, data=data)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
