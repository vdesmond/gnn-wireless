from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 16})
UPPER_LIM = 1.5

D2D = [10, 20, 30, 40, 50]
HSDL = [8, 12, 16, 24, 32]
AGG = ["min", "max", "mean", "conv", "attn"]

ls = ["solid", "dotted", "dashed", "solid", "dashed"]
mk = ["o", "^", "p", "s", "D"]


def plotter(
    arr,
    type,
    tv,
    la,
    label,
    title,
):

    x = []
    loss_flag = False
    for i in arr:
        df = pd.read_csv(f"./{type}_{la}_{tv}/{i}.csv")
        df.index += 1
        x.append(df)
    # TSBOARD_SMOOTHING = [0.5, 0.85, 0.99]

    smooth = []
    for i in range(5):
        smooth.append(x[i].ewm(alpha=0.3).mean())

    plt.figure(figsize=(12, 12))

    for i in range(5):
        l = smooth[i]["Value"].iloc[-1]
        plt.scatter(50, l, s=2)
        # plt.annotate(f"{l:.2}", (50, l))
        plt.plot(
            smooth[i]["Value"],
            linestyle=ls[i],
            marker=mk[i],
            linewidth=1.5,
            label=f"{label}: {arr[i]}",
        )
        if smooth[i]["Value"].iloc[0] > UPPER_LIM:
            loss_flag = True

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(la.lower().capitalize())

    if la == "LOSS":
        if loss_flag:
            plt.ylim(None, UPPER_LIM)
        lpos = "upper right"
    else:
        lpos = "lower right"

    plt.legend(loc=lpos)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{type}_{la}_{tv}.png")
    # plt.show()


plotter(
    D2D,
    "D2D",
    "TRAINING",
    "ACCURACY",
    "D2D",
    "Training Accuracy over epochs for various number of D2D pairs",
)
plotter(
    D2D,
    "D2D",
    "TRAINING",
    "LOSS",
    "D2D",
    "Training Loss over epochs for various number of D2D pairs",
)
plotter(
    D2D,
    "D2D",
    "VALIDATION",
    "ACCURACY",
    "D2D",
    "Vaidation Accuracy over epochs for various number of D2D pairs",
)
plotter(
    D2D,
    "D2D",
    "VALIDATION",
    "LOSS",
    "D2D",
    "Validation Loss over epochs for various number of D2D pairs",
)
plotter(
    HSDL,
    "HSD",
    "TRAINING",
    "ACCURACY",
    "HSD",
    "Training Accuracy over epochs for various Hidden State Dimensions",
)
plotter(
    HSDL,
    "HSD",
    "TRAINING",
    "LOSS",
    "HSD",
    "Training Loss over epochs for various Hidden State Dimensions",
)
plotter(
    HSDL,
    "HSD",
    "VALIDATION",
    "ACCURACY",
    "HSD",
    "Validation Accuracy over epochs for Hidden State Dimensions",
)
plotter(
    HSDL,
    "HSD",
    "VALIDATION",
    "LOSS",
    "HSD",
    "Validation Loss over epochs for Hidden State Dimensions",
)
plotter(
    AGG,
    "AGGREGATION",
    "TRAINING",
    "ACCURACY",
    "AGG",
    "Training Accuracy over epochs for various aggregation methods",
)
plotter(
    AGG,
    "AGGREGATION",
    "TRAINING",
    "LOSS",
    "AGG",
    "Training Loss over epochs for various aggregation methods",
)
plotter(
    AGG,
    "AGGREGATION",
    "VALIDATION",
    "ACCURACY",
    "AGG",
    "Validation Accuracy over epochs for various aggregation methods",
)
plotter(
    AGG,
    "AGGREGATION",
    "VALIDATION",
    "LOSS",
    "AGG",
    "Validation Loss over epochs for various aggregation methods",
)
