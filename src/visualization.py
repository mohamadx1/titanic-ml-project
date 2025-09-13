import seaborn as sns
import matplotlib.pyplot as plt

def plot_survival_by_sex(df):
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title("Survival Rate by Sex")
    plt.show()
