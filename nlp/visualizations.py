from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from nltk.corpus import stopwords
import utils.utils as utilities


# load shelter data
with open("../data/dataset.json", "r") as file:
    data = file.read()
    shelter_data = json.loads(data)

# Word Count with Stop Words
word_totals = []
words_array = []
words_100 = []
for key, shelter in shelter_data.items():
    words = utilities.get_words(shelter["description"])
    words_array.append(words)
    word_totals.append(len(words))

# histogram
sns.set(rc={'figure.figsize': (15, 9)})
word_totals = np.array(word_totals)
n_bins = 200
#fig, axs = plt.subplots(1, 2, tight_layout=True)

# N is the count in each bin, bins is the lower-limit of the bin
N, bins, patches = plt.hist(word_totals, bins=n_bins)

# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# We can also normalize our inputs by the total number of counts
# axs[1].hist(word_totals, bins=n_bins, density=True)

# Now we format the y-axis to display percentage
# axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
print("Max", np.max(word_totals))
print("Min", np.min(word_totals))
print("AVG", np.mean(word_totals))
plt.xlabel("Number of Words in Details Section")
plt.ylabel("Density")
plt.title("Histogram of Word Counts")
plt.savefig("../words_histogram.png")
plt.close()

# line graph for words
# word_totals.sort()
# plt.plot(word_totals)
# plt.xlabel("Shelter Entry")
# plt.ylabel("Number of Words")
# plt.title("Ascending Order of Words per Shelter")
# plt.savefig("../words.png")
# plt.close()

# Remove stop words
stops_words = set(stopwords.words("english"))
word_totals = []
for words in words_array:
    totals = 0
    for word in words:
        if word not in stops_words:
            totals += 1
    word_totals.append(totals)

# histogram without stop words
sns.set(rc={'figure.figsize': (15, 9)})
word_totals = np.array(word_totals)
n_bins = 120

# N is the count in each bin, bins is the lower-limit of the bin
N, bins, patches = plt.hist(word_totals, bins=n_bins)

# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

print("Max", np.max(word_totals))
print("Min", np.min(word_totals))
print("AVG", np.mean(word_totals))
plt.xlabel("Number of Words in Details Section")
plt.ylabel("Density")
plt.title("Histogram of Word Counts (Stop Words Removed)")
plt.savefig("../words_histogram_exclude_stop.png")
plt.close()


