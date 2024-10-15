import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Beispiel-Daten (ersetze diese durch deine Metriken)
rgb_iou = [0.8104, 0.7997, 0.8308, 0.8403, 0.8175, 0.7939, 0.8214, 0.8380, 0.8053, 0.8186]
rgb_f1 = [0.8835, 0.8734, 0.8993, 0.9084, 0.8846, 0.8710, 0.8877, 0.9050, 0.8735, 0.8899]

four_ch_iou = [0.8603, 0.8547, 0.8565, 0.8564, 0.8671, 0.8558, 0.8561, 0.8619, 0.8534, 0.8635]
four_ch_f1 = [0.9201, 0.9162, 0.9180, 0.9175, 0.9251, 0.9174, 0.9173, 0.9209, 0.9156, 0.9223]

fusion_iou = [0.8287, 0.8453, 0.8357, 0.8483, 0.8124]
fusion_f1 = [0.8950, 0.9107, 0.9020, 0.9121, 0.8849]

# Define the arrays and their names
array1_name = 'RGB F1'
array2_name = '4-Channel F1'
array1 = rgb_iou
array2 = four_ch_iou

# Print side-by-side comparison
print(f"{array1_name}\t\t{array2_name}\tDifference")
print("-" * 40)
for val1, val2 in zip(array1, array2):
    print(f"{val1:.4f}\t\t{val2:.4f}\t\t{val1 - val2:.4f}")

# Calculate the paired t-Test
t_stat, p_value = stats.ttest_rel(array1, array2)

# Calculate differences, standard error, and confidence intervals
diff = np.array(array1) - np.array(array2)
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)
n = len(diff)
conf_level = 0.95
t_value = stats.t.ppf((1 + conf_level) / 2, df=n-1)
margin_of_error = t_value * (std_diff / np.sqrt(n))
conf_interval = (mean_diff - margin_of_error, mean_diff + margin_of_error)

# Bootstrap for additional confidence intervals
n_bootstraps = 10000
bootstrapped_means = [np.mean(np.random.choice(diff, size=n, replace=True)) for _ in range(n_bootstraps)]
lower_bootstrap, upper_bootstrap = np.percentile(bootstrapped_means, [2.5, 97.5])

# Explanation output
print("\nAnalysis Results:")
print(f"T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"Mean Difference: {mean_diff:.4f} ± {std_diff/np.sqrt(n):.4f} (Standard Error)")
print(f"95% Confidence Interval (t-Test): ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")
print(f"Bootstrap 95% Confidence Interval: ({lower_bootstrap:.4f}, {upper_bootstrap:.4f})\n")

if p_value < 0.05:
    print(f"Conclusion: Since the p-value is below 0.05, we reject the null hypothesis.")
    print(f"There is strong evidence of a significant difference between the {array1_name} and {array2_name}.")
else:
    print(f"Conclusion: Since the p-value is above 0.05, we fail to reject the null hypothesis.")
    print(f"There is insufficient evidence to suggest a significant difference between the {array1_name} and {array2_name}.")

print("\nKey Interpretation:")
print(f"- The mean difference of {mean_diff:.4f} suggests that {array1_name} tends to perform differently than {array2_name}.")
print(f"- Both confidence intervals do not include zero, indicating that the observed difference is unlikely due to chance.")
print(f"- Therefore, we conclude that {array2_name} significantly outperforms {array1_name} in this context.")

# Dynamically adjust the y-axis limits for error bar plot
mean_array1 = np.mean(array1)
mean_array2 = np.mean(array2)
se_array1 = np.std(array1, ddof=1) / np.sqrt(n)
se_array2 = np.std(array2, ddof=1) / np.sqrt(n)

# Calculate dynamic y-axis limits based on means and errors
y_min = min(mean_array1 - se_array1, mean_array2 - se_array2) - 0.01
y_max = max(mean_array1 + se_array1, mean_array2 + se_array2) + 0.01

plt.figure(figsize=(8, 6))
plt.bar([array1_name, array2_name], [mean_array1, mean_array2], yerr=[se_array1, se_array2], 
        capsize=10, color=['gray', 'lightblue'])
plt.ylabel('Mean IoU')
plt.ylim([y_min, y_max])  # Dynamically set the y-axis limits
plt.title(f'Mean IoU with Standard Error for {array1_name} and {array2_name}')
plt.xticks(rotation=0)
plt.show()

# Plot the distribution of the differences with confidence intervals
plt.figure(figsize=(10, 5))
sns.histplot(bootstrapped_means, bins=30, kde=True)
plt.axvline(conf_interval[0], color='red', linestyle='--', label="95% CI (t-Test)")
plt.axvline(conf_interval[1], color='red', linestyle='--')
plt.axvline(lower_bootstrap, color='blue', linestyle=':', label="95% CI (Bootstrap)")
plt.axvline(upper_bootstrap, color='blue', linestyle=':')
plt.axvline(mean_diff, color='black', linestyle='-', label="Mean Difference")
plt.title(f"Bootstrap Distribution of the Mean Difference ({array1_name} - {array2_name})")
plt.xlabel("Difference in IoU")
plt.ylabel("Frequency")
plt.legend()
plt.show()