import matplotlib.pyplot as plt
import json

models = ["2e-3", "2e-4", "2e-5", "2e-6", "2e-7"]
results = [
  {
    "exact": 11.2,
    "f1": 11.828833103289625,
    "total": 250,
    "HasAns_exact": 0.6993006993006993,
    "HasAns_f1": 1.798659271485359,
    "HasAns_total": 143,
    "NoAns_exact": 25.233644859813083,
    "NoAns_f1": 25.233644859813083,
    "NoAns_total": 107
  },
{
  "exact": 32.4,
  "f1": 43.00991797515356,
  "total": 250,
  "HasAns_exact": 47.55244755244755,
  "HasAns_f1": 66.1012552013174,
  "HasAns_total": 143,
  "NoAns_exact": 12.149532710280374,
  "NoAns_f1": 12.149532710280374,
  "NoAns_total": 107
},
{
  "exact": 42.0,
  "f1": 51.44332545907123,
  "total": 250,
  "HasAns_exact": 62.93706293706294,
  "HasAns_f1": 79.44637318019447,
  "HasAns_total": 143,
  "NoAns_exact": 14.018691588785046,
  "NoAns_f1": 14.018691588785046,
  "NoAns_total": 107
},
{
  "exact": 34.0,
  "f1": 44.962923056126414,
  "total": 250,
  "HasAns_exact": 50.34965034965035,
  "HasAns_f1": 69.51559974847275,
  "HasAns_total": 143,
  "NoAns_exact": 12.149532710280374,
  "NoAns_f1": 12.149532710280374,
  "NoAns_total": 107
},
{
  "exact": 18.4,
  "f1": 30.552259151811327,
  "total": 250,
  "HasAns_exact": 31.46853146853147,
  "HasAns_f1": 52.71373977589391,
  "HasAns_total": 143,
  "NoAns_exact": 0.9345794392523364,
  "NoAns_f1": 0.9345794392523364,
  "NoAns_total": 107
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
