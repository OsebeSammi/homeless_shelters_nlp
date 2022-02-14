import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


sns.set(rc={'figure.figsize': (20, 9)})
# bar graph states
with open("../data/clean/state_shelters.json", "r") as file:
    states = json.loads(file.read())

with open("../data/clean/state_hic.json", "r") as file:
    states_hic = json.loads(file.read())

x = []
states_collected = []
states_all = []
for key, value in states.items():
    x.append(key)
    states_collected.append(value)

for key in x:
    states_all.append(states_hic[key])

# plt.bar(x, [states_all, states_collected])
# plt.xticks(rotation='vertical')
# plt.xlabel("states")
# plt.ylabel("number of homeless shelters")
# plt.title("Homeless Shelters by State")
# plt.tight_layout()
# plt.savefig("state_shelters.png")
# plt.close()

labels = x
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, states_all, width, label='HUD Inventory')
rects2 = ax.bar(x + width/2, states_collected, width, label='Crawled')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of Homeless Shelters')
plt.xlabel('States')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig("state_shelters.png")
plt.close()
