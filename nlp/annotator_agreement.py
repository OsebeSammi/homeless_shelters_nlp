import pandas as pd
import numpy as np

df_1 = pd.read_excel("../data/annotated/annotator_1.xlsx", keep_default_na=False)[100:200]
df_2 = pd.read_excel("../data/annotated/annotator_2.xlsx", keep_default_na=False)[100:200]
df_3 = pd.read_excel("../data/annotated/annotator_3.xlsx", keep_default_na=False)[100:200]
columns = ["Eligibility & Requirements", "Documents", "Services", "Admission Process", "Duration of Stay"]

annotator_agreement = []
no_infor = 0
infor = 1
infor_matches = 2
for i in range(100):
    row_agreement = [no_infor, no_infor, no_infor]
    for label in columns:
        if len(df_1.iloc[i][label].strip()) > 0 and (df_1.iloc[i][label].strip() == df_2.iloc[i][label].strip() == df_3.iloc[i][label].strip()):
            row_agreement = [infor_matches, infor_matches, infor_matches]
        else:
            if len(df_1.iloc[i][label]) > 0:
                row_agreement[0] = infor
            if len(df_2.iloc[i][label]) > 0:
                row_agreement[1] = infor
            if len(df_3.iloc[i][label]) > 0:
                row_agreement[2] = infor
        annotator_agreement.append(row_agreement)

df = pd.DataFrame(annotator_agreement)
df.to_csv("annotator_matrix.csv")
