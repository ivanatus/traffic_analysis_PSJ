import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import pingouin as pg
from scipy.stats import f_oneway
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import friedmanchisquare
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

video1 = pd.read_csv('video1_per_frame.csv')
video2 = pd.read_csv('video2_per_frame.csv')
video3 = pd.read_csv('video3_per_frame.csv')


people_video1 = video1.iloc[:, 1].values
people_video2 = video2.iloc[:, 1].values
people_video3 = video3.iloc[:, 1].values

bike_video1 = video1.iloc[:, 2].values
bike_video2 = video2.iloc[:, 2].values
bike_video3 = video3.iloc[:, 2].values

bus_video1 = video1.iloc[:, 3].values
bus_video2 = video2.iloc[:, 3].values
bus_video3 = video3.iloc[:, 3].values

car_video1 = video1.iloc[:, 4].values
car_video2 = video2.iloc[:, 4].values
car_video3 = video3.iloc[:, 4].values

truck_video1 = video1.iloc[:, 5].values
truck_video2 = video2.iloc[:, 5].values
truck_video3 = video3.iloc[:, 5].values

#Shapiro-Wilk for normality check
people_stats_video1, people_p_value_video1 = shapiro(people_video1)
people_stats_video2, people_p_value_video2 = shapiro(people_video2)
people_stats_video3, people_p_value_video3 = shapiro(people_video3)
print('\n')
print(f'Video 1, Shapiro-Wilk test on people data: p-value={people_p_value_video1}')
print(f'Video 2, Shapiro-Wilk test on people data: p-value={people_p_value_video2}')
print(f'Video 3, Shapiro-Wilk test on people data: p-value={people_p_value_video3}')

bike_stats_video1, bike_p_value_video1 = shapiro(bike_video1)
bike_stats_video2, bike_p_value_video2 = shapiro(bike_video2)
bike_stats_video3, bike_p_value_video3 = shapiro(bike_video3)

print(f'Video 1, Shapiro-Wilk test on bike data: p-value={bike_p_value_video1}')
print(f'Video 2, Shapiro-Wilk test on bike data: p-value={bike_p_value_video2}')
print(f'Video 3, Shapiro-Wilk test on bike data: p-value={bike_p_value_video3}')

bus_stats_video1, bus_p_value_video1 = shapiro(bus_video1)
bus_stats_video2, bus_p_value_video2 = shapiro(bus_video2)
bus_stats_video3, bus_p_value_video3 = shapiro(bus_video3)

print(f'Video 1, Shapiro-Wilk test on bus data: p-value={bus_p_value_video1}')
print(f'Video 2, Shapiro-Wilk test on bus data: p-value={bus_p_value_video2}')
print(f'Video 3, Shapiro-Wilk test on bus data: p-value={bus_p_value_video3}')

car_stats_video1, car_p_value_video1 = shapiro(car_video1)
car_stats_video2, car_p_value_video2 = shapiro(car_video2)
car_stats_video3, car_p_value_video3 = shapiro(car_video3)

print(f'Video 1, Shapiro-Wilk test on car data: p-value={car_p_value_video1}')
print(f'Video 2, Shapiro-Wilk test on car data: p-value={car_p_value_video2}')
print(f'Video 3, Shapiro-Wilk test on car data: p-value={car_p_value_video3}')

truck_stats_video1, truck_p_value_video1 = shapiro(truck_video1)
truck_stats_video2, truck_p_value_video2 = shapiro(truck_video2)
truck_stats_video3, truck_p_value_video3 = shapiro(truck_video3)

print(f'Video 1, Shapiro-Wilk test on truck data: p-value={truck_p_value_video1}')
print(f'Video 2, Shapiro-Wilk test on truck data: p-value={truck_p_value_video2}')
print(f'Video 3, Shapiro-Wilk test on truck data: p-value={truck_p_value_video3}')
print('\n')

print("CAR STATISTICS\n")
"""if len(car_video1) != len(car_video2) or len(car_video1) != len(car_video3) or len(car_video2) != len(car_video3):
    if len(car_video1) > len(car_video2) and len(car_video1) > len(car_video3):
        while len(car_video2) < len(car_video1):
            car_video2.append(0)
        while len(car_video3) < len(car_video1):
            car_video3.append(0)
    elif len(car_video2) > len(car_video1) and len(car_video2) > len(car_video3):
        while len(car_video1) < len(car_video2):
            car_video1.append(0)
        while len(car_video3) < len(car_video2):
            car_video3.append(0)
    else:
        while len(car_video1) < len(car_video3):
            car_video1.append(0)
        while len(car_video2) < len(car_video3):
            car_video2.append(0)"""

if len(car_video1) != len(car_video2) or len(car_video1) != len(car_video3) or len(car_video2) != len(car_video3):
    max_len = max(len(car_video1), len(car_video2), len(car_video3))
    
    if len(car_video1) < max_len:
        car_video1 = np.append(car_video1, np.zeros(max_len - len(car_video1)))
        
    if len(car_video2) < max_len:
        car_video2 = np.append(car_video2, np.zeros(max_len - len(car_video2)))
        
    if len(car_video3) < max_len:
        car_video3 = np.append(car_video3, np.zeros(max_len - len(car_video3)))

car_data = pd.DataFrame({'Video1': car_video1, 'Video2': car_video2, 'Video3': car_video3})

mauchly_result = pg.sphericity(car_data)
print("Mauchly's test for sphericity:")
print(mauchly_result)
print('\n')

levene_result = levene(car_video1, car_video2, car_video3)
print(levene_result)
print('\n')

# One-way ANOVA if data is normally distributed
if car_p_value_video1  > 0.05 and car_p_value_video2 > 0.05 and car_p_value_video2 > 0.05 and mauchly_result['spher'] and levene_result.pvalue > 0.05:
    print("ANOVA test")
    statistic, p_value = f_oneway(car_video1, car_video2, car_video3)
    print(f'F-statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The ANOVA test indicates that there is no statistically significant difference between mean values of number of cars between videos.")
# Friedman test if data not normally distributed
else:
    print("Friedman test")
    statistic, p_value = friedmanchisquare(car_video1, car_video2, car_video3)
    print(f'Statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The Friedman test indicates that there is no statistically significant difference in the mean values of the number of cars between videos.")

#post-hoc
if p_value < 0.05:
    print("------------------------------------------------------------")
    print("Post-HOC of ANOVA/Friedman")
    if car_p_value_video1 > 0.05 and car_p_value_video2 > 0.05:
        t_stasts, t_p_value = ttest_rel(car_video1, car_video2)
        print("Paired t-test on car data from video 1 and 2:")
        print(f'T-statistic: {t_stasts}')
        print("Degrees of freedom: " + string(len(car_video2) - 1))
        print(f'P-value: {t_p_value}')
        if(t_p_value > 0.05):
            print("The paired t-test indicates that there is no significant difference in mean values of number of cars between video 1 and 2.")
        if t_p_value <= 0.05:
            if(np.mean(car_video1) > np.mean(car_video2)):
                print("The paired t-test suggests a statistically significant difference in mean values of number of cars between video 1 and 2, with video 1 having higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of number of cars between video 1 and 2, with video 2 group having higher mean value.")
    else:
        w_stasts, w_p_value = wilcoxon(car_video1, car_video2)
        print("Wilcoxon Signed Rank test on car data from video 1 and 2:")
        print(f'W-statistic: {w_stasts}')
        print(f'P-value: {w_p_value}')
        if(w_p_value > 0.05):
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of number of cars between video 1 and 2.")
        if w_p_value <= 0.05:
            if np.mean(car_video1) > np.mean(car_video2):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of cars between video 1 and 2, with video 1 having higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of cars between video 1 and 2, with video 2 having higher mean value.")

    print('\n')

    if car_p_value_video1 > 0.05 and car_p_value_video2 > 0.05:
        t_stats, t_p_value = ttest_rel(car_video1, car_video2)
        print("Paired t-test on car data from video 1 and 3:")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(car_video2) - 1))
        print(f'P-value: {t_p_value}')
        if(t_p_value > 0.05):
            print("The paired t-test indicates that there is no significant difference in mean values of number of cars between video 1 and 3.")
        if t_p_value <= 0.05:
            if(np.mean(car_video1) > np.mean(car_video2)):
                print("The paired t-test suggests a statistically significant difference in mean values of number of cars between video 1 and 3, with video 1 having higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of number of cars between video 1 and 3, with video 3 having higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(car_video1, car_video3)
        print("Wilcoxon Signed Rank test on car data from video 1 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if(w_p_value > 0.05):
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of number of cars between video 1 and 3.")
        if w_p_value <= 0.05:
            if np.mean(car_video1) > np.mean(car_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of cars between video 1 and 3, with video 1 having higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of cars between video 1 and 3, with video 3 having higher mean value.")
    print('\n')

    if car_p_value_video2 > 0.05 and car_p_value_video3 > 0.05:
        t_stats, t_p_value = ttest_rel(car_video2, car_video3)
        print("Paired t-test on car data from video 2 and 3:")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(car_video2) - 1))
        print(f'P-value: {t_p_value}')
        if(t_p_value > 0.05):
            print("The paired t-test indicates that there is no significant difference in mean values of number of cars between video 2 and 3.")
        if t_p_value <= 0.05:
            if(np.mean(car_video2) > np.mean(car_video3)):
                print("The paired t-test suggests a statistically significant difference in mean values of number of cars between video 2 and 3, with video 2 having higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of number of cars between video 2 and 3, with video 3 having higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(car_video2, car_video3)
        print("Wilcoxon Signed Rank test on car data from video 2 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if(w_p_value > 0.05):
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of number of cars between video 2 and 3.")
        if w_p_value <= 0.05:
            if np.mean(car_video2) > np.mean(car_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of cars between video 2 and 3, with video 2 having higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of cars between video 2 and 3, with video 3 having higher mean value.")
print('\n')



print("PEOPLE STATISTICS\n")
if len(people_video1) != len(people_video2) or len(people_video1) != len(people_video3) or len(people_video2) != len(people_video3):
    max_len = max(len(people_video1), len(people_video2), len(people_video3))
    
    if len(people_video1) < max_len:
        people_video1 = np.append(people_video1, np.zeros(max_len - len(people_video1)))
        
    if len(people_video2) < max_len:
        people_video2 = np.append(people_video2, np.zeros(max_len - len(people_video2)))
        
    if len(people_video3) < max_len:
        people_video3 = np.append(people_video3, np.zeros(max_len - len(people_video3)))

people_data = pd.DataFrame({'Video1': people_video1, 'Video2': people_video2, 'Video3': people_video3})

# Mauchly's test for sphericity
mauchly_result = pg.sphericity(people_data)
print("Mauchly's test for sphericity:")
print(mauchly_result)
print('\n')

# Levene's test for homogeneity of variance
levene_result = levene(people_video1, people_video2, people_video3)
print("Levene's test for homogeneity of variance:")
print(levene_result)
print('\n')

# One-way ANOVA if data is normally distributed
if people_p_value_video1 > 0.05 and people_p_value_video2 > 0.05 and people_p_value_video3 > 0.05 and mauchly_result['spher'] and levene_result.pvalue > 0.05:
    print("ANOVA test")
    statistic, p_value = f_oneway(people_video1, people_video2, people_video3)
    print(f'F-statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The ANOVA test indicates that there is no statistically significant difference between mean values of number of people between videos.")
# Friedman test if data not normally distributed
else:
    print("Friedman test")
    statistic, p_value = friedmanchisquare(people_video1, people_video2, people_video3)
    print(f'Statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The Friedman test indicates that there is no statistically significant difference in the mean values of the number of people between videos.")


# Post-hoc
if p_value < 0.05:
    print("------------------------------------------------------------")
    print("Post-HOC of ANOVA/Friedman")
    
    if people_p_value_video1 > 0.05 and people_p_value_video2 > 0.05:
        t_stasts, t_p_value = ttest_rel(people_video1, people_video2)
        print("Paired t-test on people data from video 1 and 2:")
        print(f'T-statistic: {t_stasts}')
        print("Degrees of freedom: " + str(len(people_video2) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of number of people between video 1 and 2.")
        if t_p_value <= 0.05:
            if np.mean(people_video1) > np.mean(people_video2):
                print("The paired t-test suggests a statistically significant difference in mean values of number of people between video 1 and 2, with video 1 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of number of people between video 1 and 2, with video 2 having a higher mean value.")
    else:
        w_stasts, w_p_value = wilcoxon(people_video1, people_video2)
        print("Wilcoxon Signed Rank test on people data from video 1 and 2:")
        print(f'W-statistic: {w_stasts}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of number of people between video 1 and 2.")
        if w_p_value <= 0.05:
            if np.mean(people_video1) > np.mean(people_video2):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 2, with video 1 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 2, with video 2 having a higher mean value.")
    print('\n')

    if people_p_value_video1 > 0.05 and people_p_value_video3 > 0.05:
        t_stats, t_p_value = ttest_rel(people_video1, people_video3)
        print("Paired t-test on people data from video 1 and 3:")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(people_video3) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of number of people between video 1 and 3.")
        if t_p_value <= 0.05:
            if np.mean(people_video1) > np.mean(people_video3):
                print("The paired t-test suggests a statistically significant difference in mean values of number of people between video 1 and 3, with video 1 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of number of people between video 1 and 3, with video 3 having a higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(people_video1, people_video3)
        print("Wilcoxon Signed Rank test on people data from video 1 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of number of people between video 1 and 3.")
        if w_p_value <= 0.05:
            if np.mean(people_video1) > np.mean(people_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 3, with video 1 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 3, with video 3 having a higher mean value.")
    print('\n')

    if people_p_value_video2 > 0.05 and people_p_value_video3 > 0.05:
        t_stats, t_p_value = ttest_rel(people_video2, people_video3)
        print("Paired t-test on people data from video 2 and 3")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(people_video3) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of number of people between video 2 and 3.")
        if t_p_value <= 0.05:
            if np.mean(people_video2) > np.mean(people_video3):
                print("The paired t-test suggests a statistically significant difference in mean values of number of people between video 2 and 3, with video 2 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of number of people between video 2 and 3, with video 3 having a higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(people_video2, people_video3)
        print("Wilcoxon Signed Rank test on people data from video 2 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of number of people between video 2 and 3.")
        if w_p_value <= 0.05:
            if np.mean(people_video2) > np.mean(people_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 2 and 3, with video 2 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 2 and 3, with video 3 having a higher mean value.")
print('\n')


print("BIKE STATISTICS\n")

# Check and adjust lengths of arrays
if len(bike_video1) != len(bike_video2) or len(bike_video1) != len(bike_video3) or len(bike_video2) != len(bike_video3):
    max_len = max(len(bike_video1), len(bike_video2), len(bike_video3))
    
    if len(bike_video1) < max_len:
        bike_video1 = np.append(bike_video1, np.zeros(max_len - len(bike_video1)))
        
    if len(bike_video2) < max_len:
        bike_video2 = np.append(bike_video2, np.zeros(max_len - len(bike_video2)))
        
    if len(bike_video3) < max_len:
        bike_video3 = np.append(bike_video3, np.zeros(max_len - len(bike_video3)))

# Create DataFrame
bike_data = pd.DataFrame({'Video1': bike_video1, 'Video2': bike_video2, 'Video3': bike_video3})

# Mauchly's test for sphericity
mauchly_result_bike = pg.sphericity(bike_data)
print("Mauchly's test for sphericity:")
print(mauchly_result_bike)
print('\n')

# Levene's test for homogeneity of variance
levene_result_bike = levene(bike_video1, bike_video2, bike_video3)
print("Levene's test for homogeneity of variance:")
print(levene_result_bike)
print('\n')

# One-way ANOVA if data is normally distributed
#if bike_p_value_video1 > 0.05 and bike_p_value_video2 > 0.05 and bike_p_value_video3 > 0.05 and mauchly_result_bike['spher'] and levene_result_bike.pvalue > 0.05:
if bike_p_value_video1 > 0.05 and bike_p_value_video2 > 0.05 and bike_p_value_video3 > 0.05 and mauchly_result_bike.spher and levene_result_bike.pvalue > 0.05:
    print("ANOVA test")
    statistic, p_value = f_oneway(bike_video1, bike_video2, bike_video3)
    print(f'F-statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The ANOVA test indicates that there is no statistically significant difference between mean values of the number of bikes between videos.")
# Friedman test if data not normally distributed
else:
    print("Friedman test")
    statistic, p_value = friedmanchisquare(bike_video1, bike_video2, bike_video3)
    print(f'Statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The Friedman test indicates that there is no statistically significant difference in the mean values of the number of bikes between videos.")

# Post-hoc
if p_value < 0.05:
    print("------------------------------------------------------------")
    print("Post-HOC of ANOVA/Friedman")
    
    if bike_p_value_video1 > 0.05 and bike_p_value_video2 > 0.05:
        t_stats, t_p_value = ttest_rel(bike_video1, bike_video2)
        print("Paired t-test on bike data from video 1 and 2:")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(bike_video2) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of number of bikes between video 1 and 2.")
        if t_p_value <= 0.05:
            if np.mean(bike_video1) > np.mean(bike_video2):
                print("The paired t-test suggests a statistically significant difference in mean values of number of bikes between video 1 and 2, with video 1 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of number of bikes between video 1 and 2, with video 2 having a higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(bike_video1, bike_video2)
        print("Wilcoxon Signed Rank test on bike data from video 1 and 2:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of number of bikes between video 1 and 2.")
        if w_p_value <= 0.05:
            if np.mean(bike_video1) > np.mean(bike_video2):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of bikes between video 1 and 2, with video 1 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of bikes between video 1 and 2, with video 2 having a higher mean value.")
    print('\n')

    if bike_p_value_video1 > 0.05 and bike_p_value_video3 > 0.05:
        t_stats, t_p_value = ttest_rel(bike_video1, bike_video3)
        print("Paired t-test on bike data from video 1 and 3:")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(bike_video3) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of number of bikes between video 1 and 3.")
        if t_p_value <= 0.05:
            if np.mean(bike_video1) > np.mean(bike_video3):
                print("The paired t-test suggests a statistically significant difference in mean values of number of bikes between video 1 and 3, with video 1 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of number of bikes between video 1 and 3, with video 3 having a higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(bike_video1, bike_video3)
        print("Wilcoxon Signed Rank test on bike data from video 1 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of number of bikes between video 1 and 3.")
        if w_p_value <= 0.05:
            if np.mean(bike_video1) > np.mean(bike_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of bikes between video 1 and 3, with video 1 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of bikes between video 1 and 3, with video 3 having a higher mean value.")
    print('\n')

    if bike_p_value_video2 > 0.05 and bike_p_value_video3 > 0.05:
        t_stats, t_p_value = ttest_rel(bike_video2, bike_video3)
        print("Paired t-test on bike data from video 2 and 3")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(bike_video3) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of number of bikes between video 2 and 3.")
        if t_p_value <= 0.05:
            if np.mean(bike_video2) > np.mean(bike_video3):
                print("The paired t-test suggests a statistically significant difference in mean values of number of bikes between video 2 and 3, with video 2 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of number of bikes between video 2 and 3, with video 3 having a higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(bike_video2, bike_video3)
        print("Wilcoxon Signed Rank test on bike data from video 2 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of number of bikes between video 2 and 3.")
        if w_p_value <= 0.05:
            if np.mean(bike_video2) > np.mean(bike_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of bikes between video 2 and 3, with video 2 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of bikes between video 2 and 3, with video 3 having a higher mean value.")
print('\n')


print("BUS STATISTICS\n")

# Check and adjust lengths of arrays
if len(bus_video1) != len(bus_video2) or len(bus_video1) != len(bus_video3) or len(bus_video2) != len(bus_video3):
    max_len = max(len(bus_video1), len(bus_video2), len(bus_video3))
    
    if len(bus_video1) < max_len:
        bus_video1 = np.append(bus_video1, np.zeros(max_len - len(bus_video1)))
        
    if len(bus_video2) < max_len:
        bus_video2 = np.append(bus_video2, np.zeros(max_len - len(bus_video2)))
        
    if len(bus_video3) < max_len:
        bus_video3 = np.append(bus_video3, np.zeros(max_len - len(bus_video3)))

bus_data = pd.DataFrame({'Video1': bus_video1, 'Video2': bus_video2, 'Video3': bus_video3})

# Mauchly's test for sphericity
mauchly_result = pg.sphericity(bus_data)
print("Mauchly's test for sphericity:")
print(mauchly_result)
print('\n')

# Levene's test for homogeneity of variance
levene_result = levene(bus_video1, bus_video2, bus_video3)
print("Levene's test for homogeneity of variance:")
print(levene_result)
print('\n')

# One-way ANOVA if data is normally distributed
if bus_p_value_video1 > 0.05 and bus_p_value_video2 > 0.05 and bus_p_value_video3 > 0.05 and mauchly_result['spher'] and levene_result.pvalue > 0.05:
    print("ANOVA test")
    statistic, p_value = f_oneway(bus_video1, bus_video2, bus_video3)
    print(f'F-statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The ANOVA test indicates that there is no statistically significant difference between mean values of the number of people between videos.")
# Friedman test if data not normally distributed
else:
    print("Friedman test")
    statistic, p_value = friedmanchisquare(bus_video1, bus_video2, bus_video3)
    print(f'Statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The Friedman test indicates that there is no statistically significant difference in the mean values of the number of people between videos.")

# Post-hoc
if p_value < 0.05:
    print("------------------------------------------------------------")
    print("Post-HOC of ANOVA/Friedman")
    
    if bus_p_value_video1 > 0.05 and bus_p_value_video2 > 0.05:
        t_stasts, t_p_value = ttest_rel(bus_video1, bus_video2)
        print("Paired t-test on people data from video 1 and 2:")
        print(f'T-statistic: {t_stasts}')
        print("Degrees of freedom: " + str(len(bus_video2) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of the number of people between video 1 and 2.")
        if t_p_value <= 0.05:
            if np.mean(bus_video1) > np.mean(bus_video2):
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 1 and 2, with video 1 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 1 and 2, with video 2 having a higher mean value.")
    else:
        w_stasts, w_p_value = wilcoxon(bus_video1, bus_video2)
        print("Wilcoxon Signed Rank test on people data from video 1 and 2:")
        print(f'W-statistic: {w_stasts}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of the number of people between video 1 and 2.")
        if w_p_value <= 0.05:
            if np.mean(bus_video1) > np.mean(bus_video2):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 2, with video 1 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 2, with video 2 having a higher mean value.")
    print('\n')

    if bus_p_value_video1 > 0.05 and bus_p_value_video3 > 0.05:
        t_stats, t_p_value = ttest_rel(bus_video1, bus_video3)
        print("Paired t-test on people data from video 1 and 3:")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(bus_video3) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of the number of people between video 1 and 3.")
        if t_p_value <= 0.05:
            if np.mean(bus_video1) > np.mean(bus_video3):
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 1 and 3, with video 1 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 1 and 3, with video 3 having a higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(bus_video1, bus_video3)
        print("Wilcoxon Signed Rank test on people data from video 1 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of the number of people between video 1 and 3.")
        if w_p_value <= 0.05:
            if np.mean(bus_video1) > np.mean(bus_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 3, with video 1 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 3, with video 3 having a higher mean value.")
    print('\n')

    if bus_p_value_video2 > 0.05 and bus_p_value_video3 > 0.05:
        t_stats, t_p_value = ttest_rel(bus_video2, bus_video3)
        print("Paired t-test on people data from video 2 and 3")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(bus_video3) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of the number of people between video 2 and 3.")
        if t_p_value <= 0.05:
            if np.mean(bus_video2) > np.mean(bus_video3):
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 2 and 3, with video 2 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 2 and 3, with video 3 having a higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(bus_video2, bus_video3)
        print("Wilcoxon Signed Rank test on people data from video 2 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of the number of people between video 2 and 3.")
        if w_p_value <= 0.05:
            if np.mean(bus_video2) > np.mean(bus_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 2 and 3, with video 2 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 2 and 3, with video 3 having a higher mean value.")
print('\n')

print("TRUCK STATISTICS\n")

# Check and adjust lengths of arrays
if len(truck_video1) != len(truck_video2) or len(truck_video1) != len(truck_video3) or len(truck_video2) != len(truck_video3):
    max_len = max(len(truck_video1), len(truck_video2), len(truck_video3))
    
    if len(truck_video1) < max_len:
        truck_video1 = np.append(truck_video1, np.zeros(max_len - len(truck_video1)))
        
    if len(truck_video2) < max_len:
        truck_video2 = np.append(truck_video2, np.zeros(max_len - len(truck_video2)))
        
    if len(truck_video3) < max_len:
        truck_video3 = np.append(truck_video3, np.zeros(max_len - len(truck_video3)))

truck_data = pd.DataFrame({'Video1': truck_video1, 'Video2': truck_video2, 'Video3': truck_video3})

# Mauchly's test for sphericity
mauchly_result = pg.sphericity(truck_data)
print("Mauchly's test for sphericity:")
print(mauchly_result)
print('\n')

# Levene's test for homogeneity of variance
levene_result = levene(truck_video1, truck_video2, truck_video3)
print("Levene's test for homogeneity of variance:")
print(levene_result)
print('\n')

# One-way ANOVA if data is normally distributed
if truck_p_value_video1 > 0.05 and truck_p_value_video2 > 0.05 and truck_p_value_video3 > 0.05 and mauchly_result['spher'] and levene_result.pvalue > 0.05:
    print("ANOVA test")
    statistic, p_value = f_oneway(truck_video1, truck_video2, truck_video3)
    print(f'F-statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The ANOVA test indicates that there is no statistically significant difference between mean values of the number of people between videos.")
# Friedman test if data not normally distributed
else:
    print("Friedman test")
    statistic, p_value = friedmanchisquare(truck_video1, truck_video2, truck_video3)
    print(f'Statistic: {statistic}')
    print(f'P-value: {p_value}')
    if p_value > 0.05:
        print("The Friedman test indicates that there is no statistically significant difference in the mean values of the number of people between videos.")

# Post-hoc
if p_value < 0.05:
    print("------------------------------------------------------------")
    print("Post-HOC of ANOVA/Friedman")
    
    if truck_p_value_video1 > 0.05 and truck_p_value_video2 > 0.05:
        t_stasts, t_p_value = ttest_rel(truck_video1, truck_video2)
        print("Paired t-test on people data from video 1 and 2:")
        print(f'T-statistic: {t_stasts}')
        print("Degrees of freedom: " + str(len(truck_video2) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of the number of people between video 1 and 2.")
        if t_p_value <= 0.05:
            if np.mean(truck_video1) > np.mean(truck_video2):
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 1 and 2, with video 1 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 1 and 2, with video 2 having a higher mean value.")
    else:
        w_stasts, w_p_value = wilcoxon(truck_video1, truck_video2)
        print("Wilcoxon Signed Rank test on people data from video 1 and 2:")
        print(f'W-statistic: {w_stasts}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of the number of people between video 1 and 2.")
        if w_p_value <= 0.05:
            if np.mean(truck_video1) > np.mean(truck_video2):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 2, with video 1 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 2, with video 2 having a higher mean value.")
    print('\n')

    if truck_p_value_video1 > 0.05 and truck_p_value_video3 > 0.05:
        t_stats, t_p_value = ttest_rel(truck_video1, truck_video3)
        print("Paired t-test on people data from video 1 and 3:")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(truck_video3) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of the number of people between video 1 and 3.")
        if t_p_value <= 0.05:
            if np.mean(truck_video1) > np.mean(truck_video3):
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 1 and 3, with video 1 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 1 and 3, with video 3 having a higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(truck_video1, truck_video3)
        print("Wilcoxon Signed Rank test on people data from video 1 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of the number of people between video 1 and 3.")
        if w_p_value <= 0.05:
            if np.mean(truck_video1) > np.mean(truck_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 3, with video 1 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 1 and 3, with video 3 having a higher mean value.")
    print('\n')

    if truck_p_value_video2 > 0.05 and truck_p_value_video3 > 0.05:
        t_stats, t_p_value = ttest_rel(truck_video2, truck_video3)
        print("Paired t-test on people data from video 2 and 3")
        print(f'T-statistic: {t_stats}')
        print("Degrees of freedom: " + str(len(truck_video3) - 1))
        print(f'P-value: {t_p_value}')
        if t_p_value > 0.05:
            print("The paired t-test indicates that there is no significant difference in mean values of the number of people between video 2 and 3.")
        if t_p_value <= 0.05:
            if np.mean(truck_video2) > np.mean(truck_video3):
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 2 and 3, with video 2 having a higher mean value.")
            else:
                print("The paired t-test suggests a statistically significant difference in mean values of the number of people between video 2 and 3, with video 3 having a higher mean value.")
    else:
        w_stats, w_p_value = wilcoxon(truck_video2, truck_video3)
        print("Wilcoxon Signed Rank test on people data from video 2 and 3:")
        print(f'W-statistic: {w_stats}')
        print(f'P-value: {w_p_value}')
        if w_p_value > 0.05:
            print("The Wilcoxon Signed Rank test indicates that there is no significant difference in mean values of the number of people between video 2 and 3.")
        if w_p_value <= 0.05:
            if np.mean(truck_video2) > np.mean(truck_video3):
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 2 and 3, with video 2 having a higher mean value.")
            else:
                print("The Wilcoxon signed-rank test indicates a statistically significant difference in mean values of the number of people between video 2 and 3, with video 3 having a higher mean value.")
print('\n')
