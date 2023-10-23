

import numpy as np
import pandas as pd
import scipy.stats as stats

def statistic_std(x, axis):

    return np.std(x, axis)

def statistic_mean(x, axis):

    return np.mean(x, axis)

def statistic_median(x, axis):

    return np.median(x, axis)

def compute_monte_carlo(x,statistic,n_resamples,rvs_type = 'normal',alternative='two-sided',batch =1000,rng = np.random.default_rng()):
    if rvs_type== 'normal':
        rvs = lambda size: stats.norm.rvs(size=size, random_state=rng)
    elif rvs_type== 'students-t':
        df = 100
        rvs = lambda size: stats.t(df).rvs(size=size, random_state=rng) # Compare against student's t distribution 
    res = stats.monte_carlo_test(x, rvs, statistic, vectorized=True,alternative=alternative,n_resamples=n_resamples,batch=batch)
    return res


def get_pval_str(pv):
    if pv<=.001:
        pv_str ='***'
    elif pv<=.01:
        pv_str ='**'
    elif pv<=.05:
        pv_str ='*'
    else:
        pv_str ='n.s.'

    return pv_str

def get_tests_table(groups_test, name_preff, name_suff,restart_pd = False ,tests_pd=None,alternative='two-sided',alpha_p=.05):
    
    if restart_pd ==True:
        d= {'name_test': [], 'pvals': [],'stats':[],'stat_name':[],'alternative':[],'group_nb':[],'group_name':[],'N':[]}
        tests_pd = pd.DataFrame(data=d)

    N_n = np.asarray([[len(ig[iig]) for ig in groups_test] for iig in range(2)]).flatten()
    
    pvals_n = np.asarray([[stats.normaltest(np.asarray(ig[iig])[~np.isnan(ig[iig])]).pvalue for ig in groups_test] for iig in range(2)]).flatten()
    stats_n = np.asarray([[stats.normaltest(np.asarray(ig[iig])[~np.isnan(ig[iig])]).statistic for ig in groups_test] for iig in range(2)]).flatten()
    new_row =  pd.DataFrame({   'name_test':['normaltest'], 
                                'pvals':[pvals_n],
                                'stats':[stats_n],
                                'stat_name': ['s^2 + k^2'], 
                                'group_nb':[np.nan],
                                'group_name':[ name_preff+'_all'] ,
                                'N':[N_n]  })

    tests_pd = pd.concat([new_row,tests_pd.loc[:]]).reset_index(drop=True)

    for i_group in range(len(groups_test)):
        to_test =groups_test[i_group]
        if sum(np.asarray(pvals_n)<alpha_p) >0:  # not drawn from a normal distribution
            print("The null hypothesis can be rejected: not drawn from a normal distribution")
            if len(to_test[0])>=8 and len(to_test[1])>=8 :
                tt = stats.ranksums(np.asarray(to_test[0])[~np.isnan(to_test[0])],np.asarray(to_test[1])[~np.isnan(to_test[1])],alternative=alternative)
                new_row =  pd.DataFrame({'name_test':'ranksum', 
                                        'pvals':[tt.pvalue], 
                                        'stats':[tt.statistic],
                                        'stat_name':'U-statistic', 
                                        'group_nb':i_group,
                                        'alternative': alternative,
                                        'group_name': [name_preff+ '_'+ (name_suff[i_group])],
                                        'N':[[len(to_test[0]),len(to_test[1])] ] })
                tests_pd = pd.concat([new_row,tests_pd.loc[:]]).reset_index(drop=True)
            # Wilcoxon rank-sum test (The Mann–Whitney U test): tests the null hypothesis that 
            # two sets of measurements are drawn from the same distribution. 
            # The alternative hypothesis : values in one sample are more likely to be larger than the values in the other sample.
            # Name stats: U -statistic
            # tt = stats.kruskal(to_test[0],to_test[1],nan_policy='omit')
            # new_row =  pd.DataFrame({'name_test':'kruskal', 'pvals':[tt.pvalue], 
            #                         'stats':[tt.statistic],'stat_name':'H-statistic', 
            #                         'group_nb':i_group,
            #                         'group_name': [name_preff+ '_'+  (name_suff[i_group])] ,
            #                         'N':[[len(to_test[0]),len(to_test[1])] ] })
            # tests_pd = pd.concat([new_row,tests_pd.loc[:]]).reset_index(drop=True)
            # # The Kruskal-Wallis H-test: tests the null hypothesis that the population median of 
            # # all of the groups are equal.
            # # Name stats: Kruskal-Wallis H statistic
            # # tt =stats.wilcoxon(to_test[0],to_test[1]) 
            # # Wilcoxon signed-rank test tests the null hypothesis that two related paired 
            # # samples come from the same distribution: it tests whether the distribution of the differences
            # #  x - y is symmetric about zero. 
            # # It is a non-parametric version of the paired T-test.
            # # **** NOT applied because: The Mann–Whitney U test is applied to independent samples.
            # # The Wilcoxon signed-rank test is applied to matched or dependent samples. **** 
            # pvals_n = np.asarray([[stats.normaltest(ig[iig][~np.isnan(ig[iig])]).pvalue for ig in groups_test] for iig in range(2)]).flatten()
        else:
            print("The null hypothesis cannot be rejected") 
            tt = stats.ttest_ind(np.asarray(to_test[0])[~np.isnan(to_test[0])],np.asarray(to_test[1])[~np.isnan(to_test[1])],alternative=alternative)
            new_row =  pd.DataFrame({'name_test':'ttest', 'pvals':[tt.pvalue], 
                                    'stats':[tt.statistic],'stat_name':'t-statistic', 
                                    'alternative': alternative,
                                    'group_nb':i_group,'group_name': [name_preff + '_'+  (name_suff[i_group])],
                                    'N':[[len(to_test[0]),len(to_test[1])] ] })
            tests_pd = pd.concat([new_row,tests_pd.loc[:]]).reset_index(drop=True)
    
    return tests_pd


def calculate_kinkiness(c10, c50, c90, typeMedian):
    nCells = c10.shape[0]
    
    if typeMedian =='both':
        cueKinkiness = np.nan*np.ones((nCells,2))
        kinkPvals =  np.nan*np.ones((nCells,2))
    else:
        cueKinkiness = np.nan*np.ones((nCells,1))
        kinkPvals =  np.nan*np.ones((nCells,1))

    for iCell in range(nCells):
        if typeMedian =='normalized':
            diffFromInterp =  np.divide((c50[iCell,:] - np.nanmean(c10[iCell,:])) ,(np.nanmean(c90[iCell,:]) - np.nanmean(c10[iCell,:]) ) )#   % use the interpolated 50%
            tt = stats.ttest_1samp(diffFromInterp, 0)
            cueKinkiness[iCell] = tt.statistic
            kinkPvals[iCell] = tt.pvalue
            medNames = 'empirical'
        elif typeMedian =='interpolated':
            diffFromInterp = c50[iCell,:] - (np.nanmean(c10[iCell,:]) + np.nanmean(c90[iCell,:])) / 2  # use the interpolated 50%
            diffFromInterp = diffFromInterp[np.logical_not(np.isnan(diffFromInterp))]
            tt = stats.ttest_1samp(diffFromInterp, 0)
            cueKinkiness[iCell] = tt.statistic
            kinkPvals[iCell] = tt.pvalue
            medNames = 'interpolated'

        elif typeMedian =='both':
            diffFromInterp2 = c50[iCell,:] - (np.nanmean(c10[iCell,:]) + np.nanmean(c90[iCell,:])) / 2  # use the interpolated 50%
            diffFromInterp2 = diffFromInterp2[np.logical_not(np.isnan(diffFromInterp2))]

            diffFromInterp3 = np.divide((c50[iCell,:] - np.nanmean(c10[iCell,:])) ,(np.nanmean(c90[iCell,:]) - np.nanmean(c10[iCell,:]) ) )#   % use the interpolated 50%
            diffFromInterp3 = diffFromInterp3[np.logical_not(np.isnan(diffFromInterp3))]
            diffFromInterp3 = diffFromInterp3-0.5

            tt = stats.ttest_1samp(diffFromInterp2, 0)
            cueKinkiness[iCell,0] = tt.statistic
            kinkPvals[iCell,0] = tt.pvalue

            tt = stats.ttest_1samp(diffFromInterp3, 0)
            cueKinkiness[iCell,1] = tt.statistic
            kinkPvals[iCell,1] = tt.pvalue

            medNames =[ 'interpolated','normalized']

    return cueKinkiness,kinkPvals,medNames
