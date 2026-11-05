import numpy             as np
import matplotlib.pyplot as plt

from Methods import method_names



def metrics(scores):
    print(f"{"Метод":<15} | {"R^2":<30} | {"MAE":<30} | {"RMSE":<30} ")
    print("-" * 120)
    for i in range(len(method_names)):
        print(f"{method_names[i]:<15} | {scores[i][0]:>30} | {scores[i][1]:>30} | {scores[i][2]:>30}")



def schedule(scores):
    data = np.array(scores)

    R_2  = data[:, 0]
    MAE  = data[:, 1]
    RMSE = data[:, 2]

    x     = np.arange(len(scores))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width/2, MAE,  width, label='MAE' )
    ax.bar(x + width/2, RMSE, width, label='RMSE')
    ax.plot(x, R_2, label='R^2', color='red', linestyle='--', marker='o', linewidth=2)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()