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

# Data points for Fusion, RGB, and RGB+IR
fusion_iou_300 = [0.8456296920776367, 0.8231863975524902, 0.8323723077774048, 0.8065193891525269, 0.853040337562561, 0.779777467250824, 0.8334035873413086, 0.8227640986442566, 0.8242654204368591, 0.8218393325805664]
rgb_iou_300 = [0.8104, 0.7997, 0.8308, 0.8403, 0.8175, 0.7939, 0.8214, 0.8380, 0.8053, 0.8186]
rgb_ir_iou_300 = [0.8603353500366211, 0.8534227013587952, 0.8564544916152954, 0.8670772314071655, 0.8563562035560608, 0.8546502590179443, 0.8561155200004578, 0.8619412183761597, 0.8635213375091553, 0.8558455109596252]

fusion_f1_300 = [0.9102001190185547, 0.8916281461715698, 0.895847499370575, 0.8854640126228333, 0.9154802560806274, 0.8560321927070618, 0.9008628726005554, 0.891095757484436, 0.8930608034133911, 0.8889360427856445]
rgb_f1_300 = [0.8835, 0.8734, 0.8993, 0.9084, 0.8846, 0.8710, 0.8877, 0.9050, 0.8735, 0.8899]
rgb_ir_f1_300 = [0.9201040267944336, 0.9156062602996826, 0.917998194694519, 0.9251164197921753, 0.9174926280975342, 0.9162375330924988, 0.9173014760017395, 0.9209006428718567, 0.9222842454910278, 0.9173701405525208]

fusion_iou_150 = [0.6609821319580078, 0.6555767059326172, 0.6776677966117859, 0.6784079074859619, 0.683897852897644, 0.6895888447761536, 0.6447031497955322, 0.6817044615745544, 0.6974907517433167, 0.6930604577064514]
rgb_iou_150 = [0.6290138363838196, 0.6485245227813721, 0.6681326031684875, 0.6460736989974976, 0.6751046776771545, 0.6330831050872803, 0.640870988368988, 0.6475020051002502, 0.6310896277427673, 0.6508333683013916]
rgb_ir_iou_150 = [0.7204399108886719, 0.735909104347229, 0.7114688158035278, 0.71355140209198, 0.7210077047348022, 0.7199722528457642, 0.7161685228347778, 0.7381810545921326, 0.7403489947319031, 0.7033548951148987]

fusion_f1_150 = [0.7659578919410706, 0.7608004212379456, 0.7780500650405884, 0.7826518416404724, 0.7895084619522095, 0.784987211227417, 0.7525204420089722, 0.7802173495292664, 0.8007029891014099, 0.797812283039093]
rgb_f1_150 = [0.7375502586364746, 0.7572742700576782, 0.7749645113945007, 0.7450141906738281, 0.7838250994682312, 0.7450137138366699, 0.7496798038482666, 0.7490460276603699, 0.7351855635643005, 0.7569698095321655]
rgb_ir_f1_150 = [0.8138923645019531, 0.8293201327323914, 0.8091087937355042, 0.8094202280044556, 0.8143477439880371, 0.8197906017303467, 0.8114209771156311, 0.832054615020752, 0.8350836038589478, 0.8015888929367065]

fusion_iou_75 = [0.6304159760475159, 0.5985770225524902, 0.612906813621521, 0.6071501970291138, 0.6126234531402588, 0.6125524640083313, 0.6473525762557983, 0.5721817016601562, 0.6170665621757507, 0.6019647121429443]
rgb_iou_75 = [0.544189453125, 0.522980272769928, 0.5052087903022766, 0.436346173286438, 0.45466431975364685, 0.5541731119155884, 0.5027971267700195, 0.5072473287582397, 0.559797465801239, 0.3746106028556824]
rgb_ir_iou_75 = [0.5431749820709229, 0.5659253001213074, 0.5796657204627991, 0.6501533389091492, 0.5537291169166565, 0.6071853041648865, 0.6150622367858887, 0.5876942873001099, 0.5302245616912842, 0.5703843832015991]

fusion_f1_75 = [0.7355136275291443, 0.7128382325172424, 0.7213715314865112, 0.7047984600067139, 0.7150160074234009, 0.7146912217140198, 0.748618483543396, 0.6786147952079773, 0.7216756343841553, 0.7106720209121704]
rgb_f1_75 = [0.6437339782714844, 0.633655846118927, 0.6103895306587219, 0.5438068509101868, 0.5529637336730957, 0.6659839153289795, 0.6062508821487427, 0.6133514642715454, 0.6639139652252197, 0.46509796380996704]
rgb_ir_f1_75 = [0.6460161805152893, 0.6664665341377258, 0.6824936866760254, 0.7632781863212585, 0.6540212035179138, 0.7180535197257996, 0.7200575470924377, 0.6885603070259094, 0.6297062039375305, 0.6679180860519409]



# Combine the data into lists for easy iteration
fusion_iou = [fusion_iou_75, fusion_iou_150, fusion_iou_300]
rgb_iou = [rgb_iou_75, rgb_iou_150, rgb_iou_300]
rgb_ir_iou = [rgb_ir_iou_75, rgb_ir_iou_150, rgb_ir_iou_300]

fusion_f1 = [fusion_f1_75, fusion_f1_150, fusion_f1_300]
rgb_f1 = [rgb_f1_75, rgb_f1_150, rgb_f1_300]
rgb_ir_f1 = [rgb_ir_f1_75, rgb_ir_f1_150, rgb_ir_f1_300]

# X axis: number of images
x_axis = [75, 150, 300]

# Function for Shapiro-Wilk Test
def shapiro_test(data):
    stat, p = shapiro(data)
    return stat, p

# Calculate Shapiro-Wilk Test results for IoU and F1 data
shapiro_results = {'Fusion': {'IoU': [], 'F1': []}, 'RGB': {'IoU': [], 'F1': []}, 'RGB + IR': {'IoU': [], 'F1': []}}

# Apply Shapiro-Wilk test on each configuration
for fusion, rgb, rgbir in zip(fusion_iou, rgb_iou, rgb_ir_iou):
    shapiro_results['Fusion']['IoU'].append(shapiro_test(fusion))
    shapiro_results['RGB']['IoU'].append(shapiro_test(rgb))
    shapiro_results['RGB + IR']['IoU'].append(shapiro_test(rgbir))

for fusion, rgb, rgbir in zip(fusion_f1, rgb_f1, rgb_ir_f1):
    shapiro_results['Fusion']['F1'].append(shapiro_test(fusion))
    shapiro_results['RGB']['F1'].append(shapiro_test(rgb))
    shapiro_results['RGB + IR']['F1'].append(shapiro_test(rgbir))

# Shapiro-Wilk Test results can be printed or logged
print("Shapiro-Wilk Test Results (p-values indicate normality if > 0.05):")
print(shapiro_results)

# Create lists for averages and confidence intervals
iou_means = {'Fusion': [], 'RGB': [], 'RGB + IR': []}
f1_means = {'Fusion': [], 'RGB': [], 'RGB + IR': []}
iou_cis = {'Fusion': [], 'RGB': [], 'RGB + IR': []}
f1_cis = {'Fusion': [], 'RGB': [], 'RGB + IR': []}

# Calculate means and confidence intervals for each dataset size
for fusion, rgb, rgbir in zip(fusion_iou, rgb_iou, rgb_ir_iou):
    mean_fusion, ci_fusion = mean_confidence_interval(fusion)
    mean_rgb, ci_rgb = mean_confidence_interval(rgb)
    mean_rgbir, ci_rgbir = mean_confidence_interval(rgbir)
    iou_means['Fusion'].append(mean_fusion)
    iou_means['RGB'].append(mean_rgb)
    iou_means['RGB + IR'].append(mean_rgbir)
    iou_cis['Fusion'].append(ci_fusion)
    iou_cis['RGB'].append(ci_rgb)
    iou_cis['RGB + IR'].append(ci_rgbir)

for fusion, rgb, rgbir in zip(fusion_f1, rgb_f1, rgb_ir_f1):
    mean_fusion, ci_fusion = mean_confidence_interval(fusion)
    mean_rgb, ci_rgb = mean_confidence_interval(rgb)
    mean_rgbir, ci_rgbir = mean_confidence_interval(rgbir)
    f1_means['Fusion'].append(mean_fusion)
    f1_means['RGB'].append(mean_rgb)
    f1_means['RGB + IR'].append(mean_rgbir)
    f1_cis['Fusion'].append(ci_fusion)
    f1_cis['RGB'].append(ci_rgb)
    f1_cis['RGB + IR'].append(ci_rgbir)

# Plotting the IoU with confidence intervals
plt.figure(figsize=(10, 5))
plt.errorbar(x_axis, iou_means['Fusion'], yerr=iou_cis['Fusion'], fmt='-o', label=f'Cross Fusion', capsize=5, capthick=2)
plt.errorbar(x_axis, iou_means['RGB'], yerr=iou_cis['RGB'], fmt='-o', label=f'RGB', capsize=5, capthick=2)
plt.errorbar(x_axis, iou_means['RGB + IR'], yerr=iou_cis['RGB + IR'], fmt='-o', label=f'RGB + IR', capsize=5, capthick=2)
plt.xlabel('Anzahl Trainingsbilder')
plt.ylabel('Mean IoU')
plt.title('Mean IoU und 95%-Konfidenzintervall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting the F1 score with confidence intervals
plt.figure(figsize=(10, 5))
plt.errorbar(x_axis, f1_means['Fusion'], yerr=f1_cis['Fusion'], fmt='-o', label=f'Cross Fusion', capsize=5, capthick=2)
plt.errorbar(x_axis, f1_means['RGB'], yerr=f1_cis['RGB'], fmt='-o', label=f'RGB', capsize=5, capthick=2)
plt.errorbar(x_axis, f1_means['RGB + IR'], yerr=f1_cis['RGB + IR'], fmt='-o', label=f'RGB + IR', capsize=5, capthick=2)
plt.xlabel('Anzahl Trainingsbilder')
plt.ylabel('Mean F1 Score')
plt.title('Mean F1 mit 95%-Konfidenzintervall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
