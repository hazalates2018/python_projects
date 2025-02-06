import pandas as pd
df = pd.read_csv("C:\\Users\\hazal\\Downloads\\amazon_reviews.csv")
print(df.head())

import scipy.stats as stats
high_rated = df[df["overall"] >=4]["helpful_yes"]
low_rated= df[df["overall"] <=3]["helpful_yes"]
t_stat,p_value = stats.ttest_ind(high_rated,low_rated,equal_var=False)
print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
if p_value < 0.05:
    print("Statistically significant difference: Review rating affects helpfulness votes.")
else:
    print("No significant difference: Review rating does not affect helpfulness votes.")


df["reviewLength"] = df["reviewText"].astype(str).apply(len)
long_reviews = df[df["reviewLength"] >= 50]["helpful_yes"]
short_reviews = df[df["reviewLength"] < 50]["helpful_yes"]
print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
if p_value < 0.05:
    print("Statistically significant difference: Review length affects helpfulness votes.")
else:
    print("No significant difference: Review length does not affect helpfulness votes.")


import numpy as np
recent_reviews = df[df['day_diff'] < np.median(df['day_diff'])]['helpful_yes']
older_reviews = df[df['day_diff'] >= np.median(df['day_diff'])]['helpful_yes']
t_stat,p_value = stats.ttest_ind(recent_reviews,older_reviews,equal_var = False)
print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
if p_value < 0.05:
    print("Statistically significant difference: Review date affects helpfulness votes.")
else:
    print("No significant difference: Review date does not affect helpfulness votes.")


import numpy as np
high_vote = df[df["total_vote"] > df["total_vote"].median()]["helpful_yes"]
low_vote = df[df["total_vote"]<= df["total_vote"].median()]["helpful_yes"]
t_stat,p_value = stats.ttest_ind(high_vote,low_vote,equal_var=False)
print(f"T-Statistic:{t_stat:.4f},P-Value:{p_value:.4f}")
if p_value  < 0.05:
    print("Statistically significant difference: Review rating affects helpfulness votes.")
else:
    print("No significant difference: Review rating does not affect helpfulness votes.")

    