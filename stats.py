import scipy.stats as stats

statistic, pvalue = stats.ttest_ind_from_stats(
    0.79, 0.09, 300, 0.78, 0.03, 300, equal_var=False)

print(statistic, pvalue)