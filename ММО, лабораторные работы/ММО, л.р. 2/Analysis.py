import numpy             as np
import matplotlib.pyplot as plt

from Methods import method_names



def metrics(scores, properties_names = None):
    if properties_names is None:
        print(f"{"Метод":<15} | {"R^2":<30} | {"MAE":<30} | {"RMSE":<30} ")
        print("-" * 130)
        for i in range(len(method_names)):
            print(f"{method_names[i]:<15} | {scores[i][0]:>30} | {scores[i][1]:>30} | {scores[i][2]:>30}")
    else:
        print(f"{"Свойство":<25} | {"R^2":<30} | {"MAE":<30} | {"RMSE":<30} ")
        print("-" * 130)
        for i in range(len(properties_names)):
            print(f"{properties_names[i]:<25} | {scores[0][i]:>30} | {scores[1][i]:>30} | {scores[2][i]:>30}")



def schedule(scores, properties_names = None):
    data = np.array(scores)

    if properties_names is None:
        names = method_names
    else:
        names = properties_names
        data = data.T

    R_2  = data[:, 0]
    MAE  = data[:, 1]
    RMSE = data[:, 2]

    x     = np.arange(len(names))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width/2, MAE,  width, label='MAE' )
    ax.bar(x + width/2, RMSE, width, label='RMSE')
    ax.plot(x, R_2, label='R^2', color='red', linestyle='--', marker='o', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()