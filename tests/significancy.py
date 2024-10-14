import scipy.stats as stats

# Beispiel-Daten (ersetze diese durch deine Metriken)
model_a_scores = [0.80, 0.78, 0.81, 0.79, 0.82]
model_b_scores = [0.77, 0.76, 0.75, 0.78, 0.76]

# Berechne den gepaarten t-Test
t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
print(f"T-statistik: {t_stat}, p-Wert: {p_value}")

import numpy as np

# Berechne die Differenz der Metriken
diff = np.array(model_a_scores) - np.array(model_b_scores)
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)  # Standardabweichung der Differenz
n = len(diff)  # Stichprobengröße
conf_level = 0.95  # Konfidenzniveau (z.B. 95%)

# Berechne das Konfidenzintervall
t_value = stats.t.ppf((1 + conf_level) / 2, df=n-1)
margin_of_error = t_value * (std_diff / np.sqrt(n))
conf_interval = (mean_diff - margin_of_error, mean_diff + margin_of_error)

print(f"{conf_level*100}% Konfidenzintervall: {conf_interval}")


n_bootstraps = 10000
bootstrapped_means = []

for _ in range(n_bootstraps):
    sample = np.random.choice(diff, size=len(diff), replace=True)
    bootstrapped_means.append(np.mean(sample))

# Berechne das 95%-Konfidenzintervall
lower = np.percentile(bootstrapped_means, 2.5)
upper = np.percentile(bootstrapped_means, 97.5)
print(f"Bootstrap 95% Konfidenzintervall: ({lower}, {upper})")