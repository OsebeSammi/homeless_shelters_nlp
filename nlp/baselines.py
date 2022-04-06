import matplotlib.pyplot as plt
import json

models = ["2e-3", "2e-4", "2e-5", "2e-6", "2e-7"]
results = [
  {
  "exact": 12.0,
  "f1": 12.616333103289627,
  "total": 250,
  "HasAns_exact": 10.666666666666666,
  "HasAns_f1": 11.351481225877363,
  "HasAns_total": 225,
  "NoAns_exact": 24.0,
  "NoAns_f1": 24.0,
  "NoAns_total": 25
},
{
  "exact": 31.2,
  "f1": 41.220155773884514,
  "total": 250,
  "HasAns_exact": 34.22222222222222,
  "HasAns_f1": 45.35572863764947,
  "HasAns_total": 225,
  "NoAns_exact": 4.0,
  "NoAns_f1": 4.0,
  "NoAns_total": 25
},
{
  "exact": 40.4,
  "f1": 49.32022992446885,
  "total": 250,
  "HasAns_exact": 44.0,
  "HasAns_f1": 53.91136658274316,
  "HasAns_total": 225,
  "NoAns_exact": 8.0,
  "NoAns_f1": 8.0,
  "NoAns_total": 25
},
{
  "exact": 32.4,
  "f1": 42.475556237252746,
  "total": 250,
  "HasAns_exact": 35.55555555555556,
  "HasAns_f1": 46.750618041391945,
  "HasAns_total": 225,
  "NoAns_exact": 4.0,
  "NoAns_f1": 4.0,
  "NoAns_total": 25
},
{
  "exact": 17.6,
  "f1": 28.466820918547,
  "total": 250,
  "HasAns_exact": 19.555555555555557,
  "HasAns_f1": 31.62980102060778,
  "HasAns_total": 225,
  "NoAns_exact": 0.0,
  "NoAns_f1": 0.0,
  "NoAns_total": 25
}
]

ans_matches = []
ans_f1 = []
no_ans_f1 = []
for r in results:
  ans_matches.append(r["HasAns_exact"])
  ans_f1.append(r["HasAns_f1"])
  no_ans_f1.append(r["NoAns_f1"])

plt.plot(models, ans_matches, label="exact matches answer")
plt.plot(models, ans_f1, label="F1 score answer")
plt.plot(models, no_ans_f1, label="F1 score no answer")
plt.title("Performance for Fine-Tuned QA Models (4 training epochs)")
plt.xlabel("model learning rate")
plt.ylabel("performance")
plt.legend()
plt.savefig("roberta_performance")
