import numpy as np
import pandas as pd
from evaluate import compute_f1 as f1_score
from statsmodels.stats.inter_rater import fleiss_kappa

df_1 = pd.read_excel("../data/annotated/annotator_1.xlsx", keep_default_na=False)[100:200]
df_2 = pd.read_excel("../data/annotated/annotator_2.xlsx", keep_default_na=False)[100:200]
df_3 = pd.read_excel("../data/annotated/annotator_3.xlsx", keep_default_na=False)[100:200]
columns = ["Eligibility & Requirements", "Documents", "Services", "Admission Process", "Duration of Stay"]

annotator_agreement = []
no_infor = 0
infor = 1
boolean_annotations = []
f1_annotator = []

for i in range(100):
    annt_1 = ""
    annt_2 = ""
    annt_3 = ""
    for label in columns:
        blank = 0
        has_content = 0
        # annotator 1
        if len(df_1.iloc[i][label].strip()) > 0:
            has_content += 1
            annt_1 = df_1.iloc[i][label].strip()
        else:
            blank += 1

        # annotator 2
        if len(df_2.iloc[i][label].strip()) > 0:
            has_content += 1
            annt_2 = df_2.iloc[i][label].strip()
        else:
            blank += 1

        # annotator 3
        if len(df_3.iloc[i][label].strip()) > 0:
            has_content += 1
            annt_3 = df_3.iloc[i][label].strip()
        else:
            blank += 1

        # boolean annotation
        boolean_annotations.append([blank, has_content])

    # similarity
    annt_1 = str.lower(annt_1.strip())
    annt_2 = str.lower(annt_2.strip())
    annt_3 = str.lower(annt_3.strip())

    inter_annotator = [[0, f1_score(annt_1, annt_2), f1_score(annt_1, annt_3)],
                       [f1_score(annt_2, annt_1), 0, f1_score(annt_2, annt_3)],
                       [f1_score(annt_3, annt_1), f1_score(annt_3, annt_2), 0]]

    f1_annotator.append(inter_annotator)

# fleiss kapa
f_kappa = fleiss_kappa(boolean_annotations)
print(f_kappa)

# f1 inter annotator
f1_inter_ann = np.mean(f1_annotator, axis=0)
print(f1_inter_ann)
