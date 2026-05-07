import numpy as np
from scipy import stats

# Replace these with the actual data you collect from your 20 runs
ppo_results = [142, 150, 138, 145, 160, 141, 139, 155, 140, 144, 148, 137, 152, 146, 143, 149, 151, 136, 147, 142]
sac_results = [120, 125, 118, 130, 122, 119, 128, 121, 124, 126, 117, 129, 123, 127, 120, 125, 119, 122, 128, 121]

# Calculate Means
ppo_mean = np.mean(ppo_results)
sac_mean = np.mean(sac_results)

# Run Welch's t-test (equal_var=False makes it Welch's)
t_stat, p_value = stats.ttest_ind(ppo_results, sac_results, equal_var=False)

print(f"PPO Mean Steps: {ppo_mean:.2f}")
print(f"SAC Mean Steps: {sac_mean:.2f}")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.6f}")

if p_value < 0.05:
    print("\nConclusion: The difference IS statistically significant (Reject the null hypothesis).")
else:
    print("\nConclusion: The difference is NOT statistically significant (Fail to reject the null hypothesis).")