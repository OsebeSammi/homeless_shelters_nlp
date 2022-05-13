import matplotlib.pyplot as plt
import seaborn as sns

vanilla_loss = [
  1.478902,
  1.462058,
  1.385576,
  1.533293,
  1.572271,
  1.645058
]

masked_loss = [
  2.029829,
  1.948605,
  1.922229,
  2.059201,
  2.057773,
  2.156927
]

null_loss = [
  1.588394,
  1.468732,
  1.537416,
  1.543341,
  1.611972,
  1.613856
]

lr_04 = [
  1.831426,
  1.826412,
  1.68989,
  1.764074,
  2.427779,
  2.806204
]

lr_06 = [
  1.813094,
  1.721693,
  1.657546,
  1.653626,
  1.61155,
  1.600323,
  1.590468,
  1.572258,
  1.553018,
  1.562446,
  1.558106,
  1.561763
]

# sns.set(rc={'figure.figsize': (15, 10)})
# different approaches
epochs = [1, 2, 3, 4, 5, 6]
plt.plot(epochs, vanilla_loss, label="vanilla")
plt.plot(epochs, masked_loss, label="masked key words")
plt.plot(epochs, null_loss, label="added null samples")
plt.title("Fine Tuning Roberta Various Approaches")
plt.xlabel("epochs")
plt.ylabel("loss value")
plt.legend()
plt.savefig("roberta_performance")

plt.close()
# different learning rates
epochs_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
plt.plot(epochs, vanilla_loss, label="lr 2e-5")
plt.plot(epochs, lr_04, label="lr 2e-4")
plt.plot(epochs_2, lr_06, label="lr 2e-6")
plt.title("Fine Tuning Learning Rates")
plt.xlabel("epochs")
plt.ylabel("loss value")
plt.legend()
plt.savefig("roberta_lr")
plt.close()

no_ans = {
  "exact": 39.6,
  "f1": 48.54925934966657,
  "total": 250,
  "HasAns_exact": 58.04195804195804,
  "HasAns_f1": 73.68751634557094,
  "HasAns_total": 143,
  "NoAns_exact": 14.953271028037383,
  "NoAns_f1": 14.953271028037383,
  "NoAns_total": 107
}

over_s_2 = [
  1.643765,
  1.592917,
  1.619818,
  1.618348,
  1.629513,
  1.614921,
]

over_s_3 = [
  1.588016,
  1.571915,
  1.53689,
  1.63393,
  1.646557,
  1.632715
]

over_s_4 = [
  1.640043,
  1.531051,
  1.515908,
  1.515673,
  1.504848,
  1.552825
]

over_s_5 = [
  1.638036,
  1.56335,
  1.520628,
  1.555116,
  1.582966,
  1.59075
]

plt.plot(epochs, over_s_2, label="twice", color="r")
plt.plot(epochs, over_s_3, label="thrice")
plt.plot(epochs, over_s_4, label="four times", color="g")
plt.plot(epochs, over_s_5, label="five times", color="y")
plt.title("Oversampling Class Documents")
plt.xlabel("epochs")
plt.ylabel("validation loss")
plt.legend()
plt.savefig("roberta_oversample")
plt.close()

