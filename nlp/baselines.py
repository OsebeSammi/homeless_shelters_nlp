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

model_mask = {
  "exact": 24.0,
  "f1": 42.696728420820136,
  "total": 250,
  "HasAns_exact": 41.95804195804196,
  "HasAns_f1": 74.64463010632892,
  "HasAns_total": 143,
  "NoAns_exact": 0.0,
  "NoAns_f1": 0.0,
  "NoAns_total": 107
}

model_no_ans = {
  "exact": 36.8,
  "f1": 45.66589539512453,
  "total": 250,
  "HasAns_exact": 54.54545454545455,
  "HasAns_f1": 70.04527166979815,
  "HasAns_total": 143,
  "NoAns_exact": 13.08411214953271,
  "NoAns_f1": 13.08411214953271,
  "NoAns_total": 107
}

vanilla_epoch_3 = {
  "exact": 33.2,
  "f1": 42.061092108616286,
  "total": 250,
  "HasAns_exact": 57.34265734265734,
  "HasAns_f1": 72.83407711296553,
  "HasAns_total": 143,
  "NoAns_exact": 0.9345794392523364,
  "NoAns_f1": 0.9345794392523364,
  "NoAns_total": 107
}


