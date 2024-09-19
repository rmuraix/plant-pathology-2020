import pandas as pd

s1 = pd.read_csv("submission.csv")
s2 = pd.read_csv("submission_distill.csv")
s3 = s1.copy()
s3.iloc[:, 1:] = (s1.iloc[:, 1:] + s2.iloc[:, 1:]) / 2
s3.to_csv("submission_mix.csv", index=False)
