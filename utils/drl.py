import numpy as np
from scipy import stats
from cProfile import label
from cmath import nan
from tabnanny import verbose
from turtle import end_fill
from scipy import interpolate
from rl.agent import *
from sklearn.linear_model import LinearRegression
import random

import statsmodels.api as sm

def compute_vi_closed_form(asym,p,r1,r2):
    vi =  (((asym)/(1-asym)) * (p/(1-p))*r1 + r2)/ (((asym)/(1-asym)) * (p/(1-p)) + 1)
    return vi



def get_pos_neg_regime_variables(zcr_i,mu_da,mu_rec_d1,mu_rec_d2,truncate_reg=True):
    # zcr_i: zero crossing point 
    id_neg = np.intp(np.arange(zcr_i))
    id_pos = np.intp(np.arange(zcr_i,6))
    da_pos = mu_da[id_pos,:].flatten()
    rec_pos = mu_rec_d1[id_pos,:].flatten()
    da_neg = mu_da[id_neg,:].flatten()
    rec_neg = mu_rec_d2[id_neg,:].flatten()
    if truncate_reg:
        id_keep_pos = (~np.isnan(da_pos))*(~np.isnan(rec_pos))*((rec_pos)>0)*(da_pos>0)
        id_keep_neg = (~np.isnan(da_neg))*(~np.isnan(rec_neg))*((rec_neg)<0)*(da_neg<0)
    else:
        id_keep_pos = (~np.isnan(da_pos))*(~np.isnan(rec_pos))
        id_keep_neg = (~np.isnan(da_neg))*(~np.isnan(rec_neg))
    da_pos = da_pos[id_keep_pos]
    da_neg = da_neg[id_keep_neg]
    rec_neg = rec_neg[id_keep_neg]
    rec_pos = rec_pos[id_keep_pos]

    return da_pos,da_neg,rec_pos,rec_neg

def extract_metrics(m,utility_axis,rpes_x,discard_nan=True):
    taus_ = np.asarray([im[8] for im in m])
    alpha_pos = np.asarray([im[2] for im in m])
    alpha_neg = np.asarray([im[6] for im in m]) 
    zcr = [im[-1] for im in m]
    zcross_util = np.asarray([z[0][0] for z in zcr])
    zcross_rpe =  np.asarray([np.interp(zi, utility_axis[ii], rpes_x) for (ii,zi) in enumerate(zcross_util)])
    
    taus_[(taus_>1)+(taus_<0)]=np.nan
    i_disc = np.argwhere(np.isnan(taus_)).flatten()
    if discard_nan:
        zcross_util=zcross_util[~np.isnan(taus_)]
        zcross_rpe=zcross_rpe[~np.isnan(taus_)]
        alpha_neg=alpha_neg[~np.isnan(taus_)]
        alpha_pos=alpha_pos[~np.isnan(taus_)]
        taus_ = taus_[~np.isnan(taus_)]

    return taus_,alpha_pos,alpha_neg,zcross_util,zcross_rpe,i_disc
# zcr = compute_zero_crossings(per_cell_i,utility_axis,1, method_zcross) 
def compute_zero_crossings(per_cell_i, utility_axis_new,n_splits, method_zcross):


    n_types_int = len(utility_axis_new)
    type_v = np.arange(n_types_int)+1
    k = np.append(0.5,0.5 + type_v)
    critvals = np.empty((len(k),n_splits))

    # # x_data = np.matlib.repmat(type_v, per_cell_i.shape[1],1).T
    x_data = np.repeat(type_v[:,np.newaxis].T, per_cell_i.shape[1],axis=0).T

    y_data = per_cell_i
    validpart = ~np.isnan(y_data)
    x_data = x_data[validpart]
    y_data = y_data[validpart]
    data2d = np.asarray([x_data,y_data]).T
    np.random.shuffle( data2d )
    splits = np.array_split(data2d, n_splits)   
    zero_crossings = np.zeros((n_splits))
    zero_crossings_y= np.zeros((n_splits))
        
    for iis in np.arange(n_splits):
        yy = splits[iis][:,1]
        xx = splits[iis][:,0]
        for ik in np.arange(len(k)):
            ix_above = xx > k[ik]
            ix_below = xx < k[ik]
            y_valid = yy
            nPosAboveZero = sum(y_valid[ix_above]>0)
            nPosBelowZero = sum(y_valid[ix_below]>0)
            nNegAboveZero = sum(y_valid[ix_above]<0)
            nNegBelowZero = sum(y_valid[ix_below]<0)
            if method_zcross=='mult':
                critvals[ik,iis]= np.divide(nPosAboveZero + nNegBelowZero,
                    nPosAboveZero + nPosBelowZero + nNegAboveZero + nNegBelowZero) 
            elif method_zcross=='count':
                critvals[ik,iis] = sum(y_valid[ix_above]>0) + sum(y_valid[ix_below]<0)
        critvals[:,iis] = 0.001*np.random.normal(size=(len(critvals[:,iis])))+critvals[:,iis]
        mcv = np.argmax(critvals[:,iis])
        zc = k[mcv]
        zero_crossings_y[iis]=zc
        if zc < 1:
            zero_crossings[iis] = utility_axis_new[0] + 0.1;       
        elif zc > n_types_int:
            zero_crossings[iis] = utility_axis_new[-1] - 0.1;      
        else:
            w = np.abs(np.diff((critvals[mcv-1,iis] ,critvals[mcv,iis] ,critvals[mcv+1,iis])))
            w = (1./w) / sum(1./w)
            zero_crossings[iis] = w[0]*utility_axis_new[mcv-1] + w[1]*utility_axis_new[mcv] 
            # zero_crossings[iis] = w[0]*utility_axis_new[int(zc-0.5)] + w[1]*utility_axis_new[int(zc+0.5)] 
        zero_crossings_mu = np.nanmean(zero_crossings)

    return zero_crossings,zero_crossings_y,zero_crossings_mu,mcv,critvals

def compute_asymmetric_scaling(utility_axis_per_cell,per_cell_i,per_cell_norm):
    n_trials = per_cell_i.shape[1]
    negpart = utility_axis_per_cell <= 0
    pospart = utility_axis_per_cell >= 0
    # neg = np.multiply(negpart, 1)
    
    # switch_id = np.argwhere(np.diff(neg)==-1)+1

    # if len(switch_id)>0:
    #     negpart[np.arange(switch_id[0]+1)] = True
    #     pospart[np.arange(switch_id[0]+1)] = False
    #     negpart[np.arange(switch_id[0],len(pospart))] = False
    #     pospart[np.arange(switch_id[0],len(pospart))] = True

    # print([negpart,pospart])
    if sum(np.multiply(negpart, 1))==1 :
    
        neg = np.multiply(negpart, 1)
        negpart[np.argwhere(neg==1).flatten()[-1]+1] = True
    if sum(np.multiply(pospart, 1))==1 :
        pos = np.multiply(pospart, 1)
        pospart[np.argwhere(pos==1).flatten()[0]-1] = True

    if  any(negpart):
        # X = np.matlib.repmat(-utility_axis_per_cell[negpart],  n_trials,1).T 
        X  = np.repeat(-utility_axis_per_cell[negpart,np.newaxis].T,n_trials,axis=0).T
        Y = -per_cell_i[negpart,:]
        
        # xNormN = np.matlib.repmat(-utility_axis_per_cell[negpart],  n_trials,1).T 
        xNormN = (np.repeat(-utility_axis_per_cell[negpart,np.newaxis].T,n_trials,axis=0).T).flatten()
        yNormN  = ( - per_cell_norm[ negpart]).flatten()
        yv = Y.flatten()
        xv = X.flatten()
        id_disc = np.isnan(yv)
        yv = np.delete(yv,id_disc)
        xv = np.delete(xv,id_disc)
        id_disc = np.isnan(yNormN)
        yNormN = np.delete(yNormN,id_disc)
        xNormN = np.delete(xNormN,id_disc)
        
        if len(np.unique(xv))>1:
            # LinearRegression
            # X ,y = xv.reshape(-1, 1) ,yv.reshape(-1, 1)
            # xNormN ,yNormN = xNormN.reshape(-1, 1) ,yNormN.reshape(-1, 1)
            # Previous using linear regression
            # coefficients, conf_int, p_val=linear_model(X,y)
            # scale_fact_neg_p = p_val
            # scale_fact_neg_SE = conf_int
            # scale_fact_neg =   coefficients[0][0]
            res_norm = stats.linregress(xNormN.flatten(),yNormN.flatten())
            scale_fact_neg_norm = res_norm.slope

            # Testing with GLM fit 
            mask = ~np.isnan(xv).any(axis=0) & ~np.isnan(yv) & ~np.isinf(xv).any(axis=0) & ~np.isinf(yv)
            xv_clean = xv[mask]
            yv_clean = yv[mask]
            X_clean, Y_clean= xv_clean.reshape(-1, 1) ,yv_clean.reshape(-1, 1)
            model = sm.GLM(Y_clean,X_clean, family=sm.families.Gaussian())

            results = model.fit()
            scale_fact_neg_p =  results.pvalues[0]
            scale_fact_neg_SE =  results.bse[0] # Standard error
            scale_fact_neg =  results.params[0]
            
            
            
        else:
            scale_fact_neg_p = np.nan
            scale_fact_neg_SE = np.nan
            scale_fact_neg = np.nan
            scale_fact_neg_norm = np.nan
    else:
        scale_fact_neg_p = np.nan
        scale_fact_neg_SE = np.nan
        scale_fact_neg = np.nan
        scale_fact_neg_norm = np.nan
            
    if any(pospart):
        X  = np.repeat(utility_axis_per_cell[pospart,np.newaxis].T,n_trials,axis=0).T
        # X = np.matlib.repmat(utility_axis_per_cell[pospart], n_trials,1 ).T 
        Y = per_cell_i[pospart,:]
        yv = Y.flatten()
        xv = X.flatten()
        
        
        id_disc = np.isnan(yv)
        yv = np.delete(yv,id_disc)
        xv = np.delete(xv,id_disc)
        # print(np.unique(xv))
        xNormP = (np.repeat(utility_axis_per_cell[pospart,np.newaxis].T,n_trials,axis=0).T).flatten()
        # xNormP = np.matlib.repmat(utility_axis_per_cell[pospart],  n_trials,1).T 
        yNormP  = (per_cell_norm[ pospart,:]).flatten()
        id_disc = np.isnan(yNormP)
        yNormP = np.delete(yNormP,id_disc)
        xNormP = np.delete(xNormP,id_disc)
        
        # print(len(np.unique(xv)))
        if len(np.unique(xv))>1:
            # X ,y = xv.reshape(-1, 1) ,yv.reshape(-1, 1)
            # xNormP ,yNormP = xNormP.reshape(-1, 1) ,yNormP.reshape(-1, 1)

            # Previous using linear regression
            # coefficients, conf_int, p_val=linear_model(X,y)
            # scale_fact_pos_p = p_val
            # scale_fact_pos_SE = conf_int
            # scale_fact_pos =  coefficients[0][0]
            res_norm = stats.linregress(xNormP.flatten(),yNormP.flatten())
            scale_fact_pos_norm = res_norm.slope

            # Testing with GLM fit 
            mask = ~np.isnan(xv).any(axis=0) & ~np.isnan(yv) & ~np.isinf(xv).any(axis=0) & ~np.isinf(yv)
            xv_clean = xv[mask]
            yv_clean = yv[mask]
            X_clean, Y_clean= xv_clean.reshape(-1, 1) ,yv_clean.reshape(-1, 1)
            
            model = sm.GLM(Y_clean,X_clean, family=sm.families.Gaussian())

            results = model.fit()
            scale_fact_pos_p =  results.pvalues[0]
            scale_fact_pos_SE =  results.bse[0] # Standard error
            scale_fact_pos =  results.params[0]


        else:
            scale_fact_pos_p = np.nan
            scale_fact_pos_SE = np.nan
            scale_fact_pos = np.nan
            scale_fact_pos_norm = np.nan
    else:
        scale_fact_pos_p = np.nan
        scale_fact_pos_SE = np.nan
        scale_fact_pos = np.nan
        scale_fact_pos_norm = np.nan
    taus = np.divide(scale_fact_pos,scale_fact_pos+scale_fact_neg)
    taus_norm = np.divide(scale_fact_pos_norm,scale_fact_pos_norm+scale_fact_neg_norm)
    # print([taus,taus_norm])
    return scale_fact_pos_p,scale_fact_pos_SE,scale_fact_pos, scale_fact_pos_norm,\
            scale_fact_neg_p,scale_fact_neg_SE,scale_fact_neg, scale_fact_neg_norm,taus,taus_norm

def linear_model(X,y):
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    # Get the coefficients and mean squared error of the residuals
    coefficients = reg.coef_
    if coefficients[0][0]<0:
        reg = LinearRegression(fit_intercept=False).fit(X, y)
    mse_resid = np.mean((reg.predict(X) - y) ** 2)
    # Calculate the standard error of each coefficient
    n,k = len(y), len(X[0])
    sse = np.sum((reg.predict(X) - y) ** 2, axis=0) / (n-k)
    se = np.array([np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                for i in range(sse.shape[0])])
    std_err = np.sqrt(mse_resid / (n - k) * np.diag(np.linalg.inv(np.dot(X.T, X))))
    # Calculate the confidence interval for each coefficient
    t_value = stats.t.ppf(1 - 0.05 / 2, n - k)
    conf_int = np.vstack((coefficients - t_value * std_err, coefficients + t_value * std_err)).T
    # Calculate pvalues 
    t_ = reg.coef_ / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_), n-k))
    
    return coefficients, conf_int, p_val

def compute_drl_metrics(per_cell_i,utility_axis,method_zcross):
    
    zcr = compute_zero_crossings(per_cell_i,utility_axis,1, method_zcross) 
    utility_axis_per_cell = utility_axis -zcr[0][0]
    # per_cell_norm = (per_cell_i - np.nanmin(per_cell_i))/(np.nanmax(per_cell_i)-np.nanmin(per_cell_i))
    per_cell_norm = (per_cell_i)/(np.nanmax(per_cell_i)-np.nanmin(per_cell_i))
    if sum(utility_axis_per_cell>0)>=1 and sum(utility_axis_per_cell<0)>=1:
        scale_fact_pos_p,scale_fact_pos_SE,scale_fact_pos, scale_fact_pos_norm,\
                    scale_fact_neg_p,scale_fact_neg_SE,scale_fact_neg, scale_fact_neg_norm,\
                    taus,taus_norm = compute_asymmetric_scaling(utility_axis_per_cell,per_cell_i,per_cell_norm)
    else:
        scale_fact_pos_p,scale_fact_pos_SE,scale_fact_pos, scale_fact_pos_norm,\
                    scale_fact_neg_p,scale_fact_neg_SE,scale_fact_neg, scale_fact_neg_norm,\
                    taus,taus_norm  = [np.nan]*10
    
    return scale_fact_pos_p,scale_fact_pos_SE,scale_fact_pos, scale_fact_pos_norm,\
                scale_fact_neg_p,scale_fact_neg_SE,scale_fact_neg, scale_fact_neg_norm,\
                taus,taus_norm,zcr

def compute_asymmetric_scaling_receptor(xx_pos,xx_pos_log,yy_pos,xx_neg,xx_neg_log,yy_neg,method_regress='siegelslopes'):
    id_keep = np.argwhere(~(np.isnan(xx_pos)+np.isnan(yy_pos))==1)[:,0]
    if len(id_keep)>2:
        if method_regress == 'siegelslopes':
            res = stats.siegelslopes(yy_pos[id_keep],xx_pos_log[id_keep])
            alpha_pos_log  = res[0]
            res = stats.siegelslopes(yy_pos[id_keep],xx_pos[id_keep] )
            alpha_pos  = res[0]
        else:
            res = stats.linregress(yy_pos[id_keep],xx_pos_log[id_keep])
            alpha_pos_log  = res.slope
            res = stats.linregress(xx_pos[id_keep], yy_pos[id_keep] )
            alpha_pos  = res.slope
    else:
        alpha_pos_log  = np.nan
        alpha_pos  = np.nan
    
    id_keep = np.argwhere(~(np.isnan(xx_neg)+np.isnan(yy_neg))==1)[:,0]
    if len(id_keep)>2:
        if method_regress == 'siegelslopes':
            res = stats.siegelslopes(yy_neg[id_keep],xx_neg_log[id_keep] )
            alpha_neg_log  = res[0]
            res = stats.siegelslopes( yy_neg[id_keep],xx_neg[id_keep])
            alpha_neg  = res[0]
        else:
            res = stats.linregress(yy_neg[id_keep],xx_neg_log[id_keep] )
            alpha_neg_log = res.slope
            res = stats.linregress(xx_neg[id_keep], yy_neg[id_keep] )
            alpha_neg  = res.slope
    else:
        alpha_neg_log  = np.nan
        alpha_neg  = np.nan
    
    tau = (alpha_pos)/(alpha_pos+alpha_neg)
    tau_log = (alpha_pos_log)/(alpha_pos_log+alpha_neg_log)

    return tau_log,tau,alpha_pos,alpha_pos_log,alpha_neg,alpha_neg_log


def compute_utility(per_cell):
    utility_axis = np.nanmean(np.nanmean(per_cell,axis=2),axis=0)
    idneg = np.argwhere(np.diff(utility_axis)<0)
    utility_axis[idneg]  =utility_axis[idneg+1]-np.random.uniform(low=.01,high=.1,size=1)

    return utility_axis

def interpolate_utility(utility_axis):
    x_ = np.arange(len(utility_axis))
    y_ = utility_axis
    x_new = np.arange(len(utility_axis)-.9,step=.1)
    f = interpolate.interp1d(x_, y_)
    utility_axis_new = f(x_new)

    return utility_axis_new

def interpolate_cell_responses(per_cell, x_old,x_new):
    n_cells = per_cell.shape[0]
    n_trials = per_cell.shape[2]

    per_cell_int = np.empty((n_cells,len(x_new),n_trials)) 
    for i_cell in np.arange(n_cells):
        per_cell_i = per_cell[i_cell,:,:]
        for it in np.arange(n_trials):
            y_ = per_cell_i[:,it]
            x_ = x_old
            f = interpolate.interp1d(x_.flatten(), y_.flatten())
            y_interp = f(x_new)
            per_cell_int[i_cell,:,it] = y_interp

    return per_cell_int

def compute_response_axes(trial_var_cells,rebase,var_name,i_group, i_var,i_win_base,i_win_resp):
    # i_win_base = 2
    # i_win_resp = iw_us
    

    if var_name == 'input_fr' or var_name == 'da_conc':
        base_cells = trial_var_cells[i_group,i_var,i_win_base,:,:,:]
        base_cells[np.isnan(base_cells)]=0
        per_cell = trial_var_cells[i_group,i_var,i_win_resp,:,:,:]
        per_cell[per_cell<=0]=np.nan
        per_cell[per_cell>500]=np.nan
        id_cells = np.argwhere(np.nansum(np.nansum(per_cell,axis=2),axis=1)>0)
        if rebase is True:
            per_cell = per_cell-base_cells

    elif var_name == 'rec_occ' :
        # d4
        i_pos = range(3,6)
        i_neg = range(3)
        trial_var_cells[trial_var_cells<0]=np.nan
        trial_var_cells[trial_var_cells>1]=np.nan
        base_cells_p = trial_var_cells[i_group,i_var[0],i_win_base,:,i_pos,:]
        base_cells_n = trial_var_cells[i_group,i_var[1],i_win_base,:,i_neg,:]
        base_cells = np.concatenate((base_cells_n,base_cells_p),axis=0)
        per_cell_p = trial_var_cells[i_group,i_var[0],i_win_resp,:,i_pos,:]
        per_cell_n = trial_var_cells[i_group,i_var[1],i_win_resp,:,i_neg,:]
        per_cell = np.concatenate((per_cell_n,per_cell_p),axis=0)
        id_cells = np.argwhere(np.nansum(np.nansum(per_cell,axis=2),axis=0)>0)
        if rebase is True:
            per_cell = per_cell-base_cells

    return per_cell, id_cells
        


if __name__ == "__main__":
    print('ss')