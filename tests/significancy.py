import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example Data (replace with your metrics)
rgb_iou = [0.8104, 0.7997, 0.8308, 0.8403, 0.8175, 0.7939, 0.8214, 0.8380, 0.8053, 0.8186]
rgb_f1 = [0.8835, 0.8734, 0.8993, 0.9084, 0.8846, 0.8710, 0.8877, 0.9050, 0.8735, 0.8899]
rgb_training_time = [267.66, 321.83, 444.03, 353.80, 316.40, 291.23, 369.68, 400.87, 327.50, 269.44]
rgb_inference_time = [0.0198, 0.0207, 0.0199, 0.0210, 0.0202, 0.0200, 0.0200, 0.0196, 0.0201, 0.0207]

four_ch_iou = [0.8603, 0.8547, 0.8565, 0.8564, 0.8671, 0.8558, 0.8561, 0.8619, 0.8534, 0.8635]
four_ch_f1 = [0.9201, 0.9162, 0.9180, 0.9175, 0.9251, 0.9174, 0.9173, 0.9209, 0.9156, 0.9223]
four_ch_training_time = [361.24, 324.14, 289.16, 281.85, 307.28, 271.90, 297.51, 353.00, 327.85, 330.47]
four_ch_inference_time = [0.0197, 0.0198, 0.0198, 0.0196, 0.0194, 0.0200, 0.0194, 0.0201, 0.0201, 0.0193]

# Define the arrays and their names
array1_name = 'RGB Training Time'
array2_name = '4-Channel Training Time'
array1 = rgb_training_time
array2 = four_ch_training_time

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

# Inference about significance
if p_value < 0.05:
    print(f"Conclusion: Since the p-value is below 0.05, we reject the null hypothesis.")
    print(f"There is strong evidence of a significant difference between {array1_name} and {array2_name}.")
else:
    print(f"Conclusion: Since the p-value is above 0.05, we fail to reject the null hypothesis.")
    print(f"There is insufficient evidence to suggest a significant difference between {array1_name} and {array2_name}.")

# Error Bar Plot with Legends for Mean and Error
mean_array1 = np.mean(array1)
mean_array2 = np.mean(array2)
se_array1 = np.std(array1, ddof=1) / np.sqrt(n)
se_array2 = np.std(array2, ddof=1) / np.sqrt(n)

# Dynamic y-axis limits based on means and errors
y_min = min(mean_array1 - se_array1, mean_array2 - se_array2) - 0.005
y_max = max(mean_array1 + se_array1, mean_array2 + se_array2) + 0.005

plt.figure(figsize=(8, 6))
bars = plt.bar([array1_name, array2_name], [mean_array1, mean_array2], 
               yerr=[se_array1, se_array2], capsize=10, color=['gray', 'lightblue'])
plt.ylabel('Mean Inference Time')
plt.ylim([y_min, y_max])  # Dynamic y-axis limits
plt.title(f'Mean Inference Time with Standard Error for {array1_name} and {array2_name}')
plt.xticks(rotation=0)

# Add error bar legend
for bar, mean, se in zip(bars, [mean_array1, mean_array2], [se_array1, se_array2]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.0005, 
             f'Mean: {mean:.4f}\nSE: ±{se:.4f}', ha='center', va='bottom', fontsize=9)

plt.legend([f'Mean ± SE'], loc='upper right')
plt.show()

# Bootstrap Distribution Plot with Confidence Interval Legends
plt.figure(figsize=(10, 5))
sns.histplot(bootstrapped_means, bins=30, kde=True)
plt.axvline(conf_interval[0], color='red', linestyle='--', label="95% CI (t-Test)")
plt.axvline(conf_interval[1], color='red', linestyle='--')
plt.axvline(lower_bootstrap, color='blue', linestyle=':', label="95% CI (Bootstrap)")
plt.axvline(upper_bootstrap, color='blue', linestyle=':')
plt.axvline(mean_diff, color='black', linestyle='-', label="Mean Difference")
plt.title(f"Bootstrap Distribution of the Mean Difference ({array1_name} - {array2_name})")
plt.xlabel("Difference in Inference Time")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.show()