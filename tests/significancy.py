import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#300 training images Dataset
rgb_iou = [0.8104, 0.7997, 0.8308, 0.8403, 0.8175, 0.7939, 0.8214, 0.8380, 0.8053, 0.8186]
rgb_f1 = [0.8835, 0.8734, 0.8993, 0.9084, 0.8846, 0.8710, 0.8877, 0.9050, 0.8735, 0.8899]
rgb_training_time = [267.66, 321.83, 444.03, 353.80, 316.40, 291.23, 369.68, 400.87, 327.50, 269.44]
rgb_inference_time = [0.0198, 0.0207, 0.0199, 0.0210, 0.0202, 0.0200, 0.0200, 0.0196, 0.0201, 0.0207]

four_ch_iou = [0.8603, 0.8547, 0.8565, 0.8564, 0.8671, 0.8558, 0.8561, 0.8619, 0.8534, 0.8635]
four_ch_f1 = [0.9201, 0.9162, 0.9180, 0.9175, 0.9251, 0.9174, 0.9173, 0.9209, 0.9156, 0.9223]
four_ch_training_time = [361.24, 324.14, 289.16, 281.85, 307.28, 271.90, 297.51, 353.00, 327.85, 330.47]
four_ch_inference_time = [0.0197, 0.0198, 0.0198, 0.0196, 0.0194, 0.0200, 0.0194, 0.0201, 0.0201, 0.0193]

fusion_IoU = [0.8287, 0.8453, 0.8357, 0.8483, 0.8124]
fusion_F1 = [0.8950, 0.9107, 0.9020, 0.9121, 0.8849]

#150 training images Dataset
rgb_IoU = [0.6290138363838196, 0.6485245227813721, 0.6681326031684875, 0.6460736989974976, 0.6751046776771545, 0.6330831050872803, 0.640870988368988, 0.6475020051002502, 0.6310896277427673, 0.6508333683013916]
rgb_F1 =  [0.7375502586364746, 0.7572742700576782, 0.7749645113945007, 0.7450141906738281, 0.7838250994682312, 0.7450137138366699, 0.7496798038482666, 0.7490460276603699, 0.7351855635643005, 0.7569698095321655]
rgb_Inference_time_per_image = [0.02424677610397339, 0.020989277362823487, 0.02009260892868042, 0.020041925907135008, 0.020074524879455567, 0.020105228424072266, 0.020140225887298583, 0.020046989917755127, 0.02009493350982666, 0.020042593479156493]

four_ch_IoU = [0.7204399108886719, 0.735909104347229, 0.7114688158035278, 0.71355140209198, 0.7210077047348022, 0.7199722528457642, 0.7161685228347778, 0.7381810545921326, 0.7403489947319031, 0.7033548951148987]
four_ch_F1 = [0.8138923645019531, 0.8293201327323914, 0.8091087937355042, 0.8094202280044556, 0.8143477439880371, 0.8197906017303467, 0.8114209771156311, 0.832054615020752, 0.8350836038589478, 0.8015888929367065]
four_ch_Inference_time_per_image = [0.02357133150100708, 0.021376008987426757, 0.02243622303009033, 0.022225794792175294, 0.021605222225189208, 0.021593306064605713, 0.021565160751342773, 0.021925745010375978, 0.02228181838989258, 0.022274136543273926]

array_names = ['RGB F1', '4-Channel F1', 'Fusion F1']
arrays = [rgb_f1, four_ch_f1, fusion_F1]

# Print side-by-side comparison
print(f"{array_names[0]}\t\t{array_names[1]}\t\t{array_names[2]}\t\tDifference (1-2)   Difference (1-3)")
print("-" * 60)
for val1, val2, val3 in zip(*arrays):
    print(f"{val1:.4f}\t\t{val2:.4f}\t\t{val3:.4f}\t\t{val1 - val2:.4f}   \t{val1 - val3:.4f}")

# Calculate paired t-Test between all combinations
for i in range(len(arrays)):
    for j in range(i + 1, len(arrays)):
        diff = np.array(arrays[i]) - np.array(arrays[j])
        t_stat, p_value = stats.ttest_rel(arrays[i], arrays[j])
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        n = len(diff)
        t_value = stats.t.ppf((1 + 0.95) / 2, df=n - 1)
        margin_of_error = t_value * (std_diff / np.sqrt(n))
        conf_interval = (mean_diff - margin_of_error, mean_diff + margin_of_error)

        print(f"\nComparison between {array_names[i]} and {array_names[j]}:")
        print(f"T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        print(f"Mean Difference: {mean_diff:.4f} ± {std_diff/np.sqrt(n):.4f}")
        print(f"95% Confidence Interval (t-Test): ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})\n")

# Error Bar Plot for all arrays
means = [np.mean(arr) for arr in arrays]
ses = [np.std(arr, ddof=1) / np.sqrt(len(arr)) for arr in arrays]
y_min = min([m - se for m, se in zip(means, ses)]) - 0.005
y_max = max([m + se for m, se in zip(means, ses)]) + 0.005

plt.figure(figsize=(8, 6))
bars = plt.bar(array_names, means, yerr=ses, capsize=10, color=['gray', 'lightblue', 'lightgreen'])
plt.ylabel('F1 Score')  # Manually setting the y-axis title
plt.ylim([y_min, y_max])
plt.title('F1 Score for 300img Dataset')

# Annotate bars with mean and standard error
for bar, mean, se in zip(bars, means, ses):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.0005,
             f'Mean: {mean:.4f}\nSE: ±{se:.4f}', ha='center', va='bottom', fontsize=9)

plt.legend([f'Mean ± SE'], loc='upper right')
plt.show()

# Bootstrap Distribution Plot for Differences between arrays
plt.figure(figsize=(15, 5))
for i in range(len(arrays) - 1):
    for j in range(i + 1, len(arrays)):
        diff = np.array(arrays[i]) - np.array(arrays[j])
        n_bootstraps = 10000
        bootstrapped_means = [np.mean(np.random.choice(diff, size=n, replace=True)) for _ in range(n_bootstraps)]
        lower_bootstrap, upper_bootstrap = np.percentile(bootstrapped_means, [2.5, 97.5])
        
        sns.histplot(bootstrapped_means, bins=30, kde=True, label=f"{array_names[i]} - {array_names[j]}")
        plt.axvline(lower_bootstrap, color='blue', linestyle=':', label=f"95% CI Bootstrap {array_names[i]} - {array_names[j]}")
        plt.axvline(upper_bootstrap, color='blue', linestyle=':')

plt.axvline(0, color='black', linestyle='-', label="Mean Difference")
plt.title("Bootstrap Distribution of Mean Differences")
plt.xlabel("Difference in Inference Time")
plt.ylabel("Frequency")
plt.legend()
plt.show()