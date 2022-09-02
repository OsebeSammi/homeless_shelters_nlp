import matplotlib.pyplot as plt
from matplotlib import pyplot as plot
import seaborn as sns
import json

sns.set(rc={'figure.figsize': (12, 9)})

with open("results/qa_1e-5", "r") as file:
    lines = file.readlines()
    lr1_epoch = []
    lr1_loss = []
    for line in lines:
        d = json.loads(line)
        lr1_epoch.append(d["epoch"])
        lr1_loss.append(d["loss"])

with open("results/qa_3e-5", "r") as file:
    lines = file.readlines()
    lr3_epoch = []
    lr3_loss = []
    for line in lines:
        d = json.loads(line)
        lr3_epoch.append(d["epoch"])
        lr3_loss.append(d["loss"])

plot.plot(lr1_epoch, lr1_loss, label="lr:1e-5")
plot.plot(lr3_epoch, lr3_loss, label="lr:3e-5")
plt.xlabel("training epochs")
plot.ylabel("training loss")
plot.title("Showing Training and Validation Loss QA")
plt.legend()
plot.show()

