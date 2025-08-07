from scipy.stats import ttest_ind, f_oneway, pearsonr

def run_ttest(group1, group2):
    return ttest_ind(group1, group2, equal_var=False)

def run_anova(groups):
    return f_oneway(*groups)

def run_correlation(x, y):
    return pearsonr(x, y)

