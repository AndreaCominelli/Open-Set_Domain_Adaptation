import math
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as npy

colors = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500", "#AB3428"]

def truncate(number, digits=3):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def plot_acc(model="obj"):
    sb.set_style("darkgrid")
    if model == "obj":
        path = "./stats/obj_stats.txt"
    else:
        path = "./stats/self_stats.txt"
    print(path)
    with open(path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(",")[0:-1]
            param = line.split(",")[-1] #This is an hyper parameter value, hence it could be whatever we want (lr, weight decay ecc..)
            epochs = []
            accuracies = []
            for d in data:
                e = d.split(":")[0]
                acc = truncate(float(d.split(":")[1]))
                epochs.append(e)
                accuracies.append(acc)
            plt
            plt.plot(epochs, accuracies, linewidth=3, color=colors[i % len(colors)], label=param)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.xticks(npy.arange(0, epochs[-1], step=10))
            plt.legend()
        plt.show()

if __name__ == "__main__":
    plot_acc(model="self")