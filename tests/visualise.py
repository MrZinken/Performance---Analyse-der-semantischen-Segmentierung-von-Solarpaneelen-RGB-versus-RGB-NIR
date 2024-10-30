import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t, shapiro

# Function to calculate mean and 95% confidence interval
def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2, n - 1)
    return mean, h

# Data for Precision values
fusion_precision_75 = [0.8299034237861633, 0.8248817920684814, 0.8212370276451111, 0.8080577850341797, 0.8069779872894287, 0.8209850192070007, 0.8369370102882385, 0.7988770008087158, 0.851389467716217, 0.8109037280082703]
rgb_precision_75 = [0.7521644830703735, 0.7705279588699341, 0.7166515588760376, 0.7235956788063049, 0.6822119951248169, 0.7305411696434021, 0.7280576229095459, 0.7221769094467163, 0.7488166093826294, 0.666259765625]
rgb_ir_precision_75 = [0.7824720144271851, 0.767415463924408, 0.7751535773277283, 0.7644031047821045, 0.7452197074890137, 0.755850613117218, 0.7737115621566772, 0.7877134680747986, 0.7486498355865479, 0.766326904296875]

fusion_precision_150 = [0.8501341938972473, 0.8574011921882629, 0.8266854286193848, 0.8545320630073547, 0.8338108062744141, 0.8429596424102783, 0.8392729163169861, 0.8472412824630737, 0.8481334447860718, 0.8390887379646301]
rgb_precision_150 = [0.8484324812889099, 0.8004934787750244, 0.855510950088501, 0.828626275062561, 0.8400769829750061, 0.8304464817047119, 0.823357105255127, 0.8210782408714294, 0.7996325492858887, 0.8245174288749695]
rgb_ir_precision_150 = [0.854154109954834, 0.841978132724762, 0.8346388339996338, 0.8509284257888794, 0.8531238436698914, 0.8427965044975281, 0.8455503582954407, 0.8537756204605103, 0.8551849126815796, 0.864866316318512]

fusion_precision_300 = [0.9361903667449951, 0.9109013676643372, 0.9381046295166016, 0.9079286456108093, 0.9290487170219421, 0.945119321346283, 0.9327863454818726, 0.896541953086853, 0.9316450953483582, 0.9071054458618164]
rgb_precision_300 = [0.9254345297813416, 0.9135575890541077, 0.9333645701408386, 0.9157202243804932, 0.9010971784591675, 0.916436493396759, 0.9204868674278259, 0.9198874235153198, 0.9267853498458862, 0.9224536418914795]
rgb_ir_precision_300 = [0.940501868724823, 0.9405944347381592, 0.9175294041633606, 0.9403547644615173, 0.9421055316925049, 0.9322125315666199, 0.9459946155548096, 0.9386456608772278, 0.9392919540405273, 0.9305249452590942]

# Combine Precision data for each model
fusion_precision = [fusion_precision_75, fusion_precision_150, fusion_precision_300]
rgb_precision = [rgb_precision_75, rgb_precision_150, rgb_precision_300]
rgb_ir_precision = [rgb_ir_precision_75, rgb_ir_precision_150, rgb_ir_precision_300]

# X axis: number of images
x_axis = [75, 150, 300]
# Function for Shapiro-Wilk Test
def shapiro_test(data):
    stat, p = shapiro(data)
    return stat, p
# Shapiro-Wilk Test for Precision
shapiro_results_precision = {'Fusion': [], 'RGB': [], 'RGB + IR': []}

for fusion, rgb, rgbir in zip(fusion_precision, rgb_precision, rgb_ir_precision):
    shapiro_results_precision['Fusion'].append(shapiro_test(fusion))
    shapiro_results_precision['RGB'].append(shapiro_test(rgb))
    shapiro_results_precision['RGB + IR'].append(shapiro_test(rgbir))

print("Shapiro-Wilk Test Results (Precision):")
print(shapiro_results_precision)

# Calculate means and confidence intervals for Precision
precision_means = {'Fusion': [], 'RGB': [], 'RGB + IR': []}
precision_cis = {'Fusion': [], 'RGB': [], 'RGB + IR': []}

for fusion, rgb, rgbir in zip(fusion_precision, rgb_precision, rgb_ir_precision):
    mean_fusion, ci_fusion = mean_confidence_interval(fusion)
    mean_rgb, ci_rgb = mean_confidence_interval(rgb)
    mean_rgbir, ci_rgbir = mean_confidence_interval(rgbir)
    precision_means['Fusion'].append(mean_fusion)
    precision_means['RGB'].append(mean_rgb)
    precision_means['RGB + IR'].append(mean_rgbir)
    precision_cis['Fusion'].append(ci_fusion)
    precision_cis['RGB'].append(ci_rgb)
    precision_cis['RGB + IR'].append(ci_rgbir)

# Plotting the Precision with confidence intervals
plt.figure(figsize=(10, 5))
plt.errorbar(x_axis, precision_means['Fusion'], yerr=precision_cis['Fusion'], fmt='-o', label=f'Fusion', capsize=5, capthick=2)
plt.errorbar(x_axis, precision_means['RGB'], yerr=precision_cis['RGB'], fmt='-o', label=f'RGB', capsize=5, capthick=2)
plt.errorbar(x_axis, precision_means['RGB + IR'], yerr=precision_cis['RGB + IR'], fmt='-o', label=f'RGB + IR', capsize=5, capthick=2)
plt.xlabel('Anzahl Trainingsbilder')
plt.ylabel('Mean Precision')
plt.title('Mean Precision mit 95%-Konfidenzintervall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

