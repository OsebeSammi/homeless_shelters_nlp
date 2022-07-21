import json
import matplotlib.pyplot as plt
from evaluate import start

# models = ["electra", "longformer", "minilm", "roberta"]
models = ["electra", "minilm", "roberta"]
counts = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]

exact_match = []
f1 = []
for model in models:
    m_em = []
    m_f1 = []
    for count in counts:
        results = start("output/gold.json", "output/"+model+count+"_pred.json")
        m_em.append(results["exact"])
        m_f1.append(results["f1"])
    exact_match.append(m_em)
    f1.append(m_f1)

samples = ["0", "50", "100", "150", "200", "250", "300", "350", "400", "450", "500"]
plt.plot(samples, exact_match[0], label="electra")
# plt.plot(samples, exact_match[1], label="longformer")
plt.plot(samples, exact_match[1], label="minilm")
plt.plot(samples, exact_match[2], label="roberta")
plt.title("Zero and Few Shot Exact Matching")
plt.xlabel("number of samples")
plt.ylabel("exact matching")
plt.legend()
plt.savefig("exact_matching")
plt.close()

plt.plot(samples, f1[0], label="electra")
# plt.plot(samples, f1[1], label="longformer")
plt.plot(samples, f1[1], label="minilm")
plt.plot(samples, f1[2], label="roberta")
plt.title("Zero and Few Shot F1")
plt.xlabel("number of samples")
plt.ylabel("F1")
plt.legend()
plt.savefig("f1")
plt.close()

