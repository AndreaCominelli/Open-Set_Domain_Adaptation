"""import numpy as npy
import seaborn as sb
import matplotlib.pyplot as plt
import random

def sinplot(flip=1):
    colors = sb.color_palette("deep")
    x = npy.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, npy.sin(x + i * .5) * (7 - i) * flip, 
                color=colors[random.randint(0, len(colors) - 1)], linewidth=3, label=f"plot {i}",
                linestyle="dotted")
    plt.legend()
    plt.xlabel("Etichetta x")
    plt.ylabel("Etichetta y")
    plt.show()

acc1 = {
    1:0.05,
    2:0.15,
    3:0.30,
    4:0.40,
    5:0.45
}

acc2={
    1:0.45,
    2:0.40,
    3:0.30,
    4:0.15,
    5:0.05
}

colors = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"]

def plot_self_acc(plot, self_accuracies, weight):
    keys = self_accuracies.keys()
    values = list(self_accuracies.values())
    plot.plot(keys, values, linewidth=3, color=colors[4], label=weight)

def plot_obj_acc():
    pass

sb.set_style("darkgrid")

fig=plt.figure()
plot = fig.add_subplot()
plot_self_acc(plot, acc1, 0.5)
plot_self_acc(plot, acc2, 0.5)
plt.show()"""