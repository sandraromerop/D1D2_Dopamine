import os 
import glob
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from utils.plots import *
from utils.drl import *
from rl.agent import *
from utils.stats_perform import *

def plot_mean_per_trial_batch_iterations(agent,results_v,vars_n,id_states,save_titles,plot_titles,y_labels,cs_names,g_names,id_cs_plot,id_iter_plot,simulation_type,fig_dir,colors_g,evtype = 'tau'):
    fig_v,ax_v = [],[]
    for iv in np.arange(len(vars_n)):
        var = vars_n[iv]
        id_state = id_states[iv]
        title_var = plot_titles[iv]
        y_title = y_labels[iv]
        save_tit = save_titles[iv]
        fig,ax = plot_var_full_normalized_iterations(agent,results_v,var,title_var,y_title,cs_names,g_names,id_state,id_iter_plot,id_cs_plot,simulation_type,save_tit,fig_dir,colors_g,evtype = evtype)
        fig_v.append(fig)
        ax_v.append(ax)
    return fig_v, ax_v

def plot_var_full_normalized_iterations(agent,results_v,var,title_var,y_title,cs_names,g_names,id_state,id_iter_plot,id_cs_plot,simulation_type,save_tit,fig_dir,colors_g,evtype = 'tau'):
    fig,ax = plt.subplots(2,2,figsize = (10,10))
    nb_cs = agent.task.nb_cs
    deliv_cs = agent.deliv_cs
    n_iterations_min = np.min([len(rr) for rr in results_v])
    n_groups = len(results_v)
    cs_mu = np.nan*np.ones((n_groups,n_iterations_min,nb_cs))
    cs_mu_n = np.nan*np.ones((n_groups,n_iterations_min,nb_cs))
    for (ir,results_group) in enumerate(results_v):
        for it in np.arange(n_iterations_min):
            id_tau = np.argmin( np.abs(results_group[it]['agent']['taus']-.5))
            results = results_group[id_iter_plot[it]]
            cs_resp = []
            for ics in np.arange(nb_cs):
                id_cs = np.argwhere(deliv_cs==ics)
                if evtype =='tau':
                    temp =results['results'][var].copy()[id_tau,id_cs,id_state,:].squeeze()
                elif evtype== 'mean':
                    temp = np.nanmean(results['results'][var].copy()[:,id_cs,id_state,:],axis=0).squeeze()
                idnan = [np.argwhere(np.isnan(temp[ics])).flatten() for ics in range(len(id_cs))]
                for ii,idn in enumerate(idnan):
                    if len(idn)==0:
                        idnan[ii] = temp.shape[1]
                    else:
                        idnan[ii] =idn[0]
                idend = np.asarray([np.max([np.min([idn-50,1000]),0]) for idn in idnan])
                temp_ = np.concatenate([temp[ics,idn:-1] for (ics,idn) in zip(range(len(id_cs)),idend)])
                temp_= temp_[~np.isnan(temp_)]
                temp_ = temp_[np.abs(temp_)<10]
                if len(temp_)>10:
                    cs_resp.append(temp_)

            if len(cs_resp) ==nb_cs:
                ll = np.min([len(ci) for ci in cs_resp])
                cs_resp_ = np.asarray([ci[:ll] for ci in cs_resp])
                cs_resp_n = (np.nanmean(cs_resp_,axis=1)-np.nanmean(cs_resp_[0,:]))/(np.nanmean(cs_resp_[2,:])-np.nanmean(cs_resp_[0,:]))
                cs_resp_n[cs_resp_n>1] = np.nan
                cs_resp_n[cs_resp_n<0] = np.nan
                cs_mu_n[ir,it,:] =cs_resp_n 
                cs_mu[ir,it,:] = [np.nanmean(_) for _ in cs_resp_]
                
        ax[0,0].errorbar(id_cs_plot,np.nanmean(cs_mu[ir,:,id_cs_plot],axis=1),np.nanstd(cs_mu[ir,:,id_cs_plot],axis=1)/np.sqrt(n_iterations_min),color=colors_g[ir])
        ax[0,1].errorbar(id_cs_plot,np.nanmean(cs_mu_n[ir,:,id_cs_plot],axis=1),np.nanstd(cs_mu_n[ir,:,id_cs_plot],axis=1)/np.sqrt(n_iterations_min),color=colors_g[ir])
        plot_config(ax[0,0],'P(r)',y_title,14,False)
        plot_config(ax[0,1],'P(r)',y_title + '(normalized)',14,False)
        xticks_(ax[0,0],id_cs_plot,cs_names)
        xticks_(ax[0,1],id_cs_plot,cs_names)
        title_(ax[0,0],title_var)
        title_(ax[0,1],title_var + 'normalized')
        list_resp = [cs_mu[ir,:,id_cs_plot[ic]] for ic in range(len(id_cs_plot)) for ir in range(2) ]
        ax[1,0].boxplot(list_resp)
        list_resp = [cs_mu_n[ir,:,1]  for ir in range(2) ]
        ax[1,1].boxplot(list_resp)
        plot_config(ax[1,0],'',y_title,14,False)
        plot_config(ax[1,1],'',y_title + '(normalized)',14,False)
        xticks_(ax[1,0],2*(id_cs_plot)+1.5,cs_names)
        xticks_(ax[1,1],1+np.arange(len(g_names)),g_names)
        title_(ax[1,0],title_var)
        title_(ax[1,1],title_var + 'normalized')
        
        

    # fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))

    return fig, ax

def get_mean_response_across_iterations(results_v,var,id_state,evtype):


    n_iterations_min = np.min([len(rr) for rr in results_v])
    n_groups = len(results_v)
    nb_cs = results_v[0][0]['task']['nb_cs']
    cs_mu = np.nan*np.ones((n_groups,n_iterations_min,nb_cs))
    cs_mu_n = np.nan*np.ones((n_groups,n_iterations_min,nb_cs))
    for (ir,results_group) in enumerate(results_v):
        id_iter_plot = np.arange(0,len(results_group))
        for it in np.arange(n_iterations_min):
            deliv_cs =  results_group[it]['task']['deliv_cs']
            id_tau = np.argmin( np.abs(results_group[it]['agent']['taus']-.5))
            results = results_group[id_iter_plot[it]]
            cs_resp = []
            for ics in np.arange(nb_cs):
                id_cs = np.argwhere(deliv_cs==ics)
                if evtype =='tau':
                    temp =results['results'][var].copy()[id_tau,id_cs,id_state,:].squeeze()
                elif evtype== 'mean':
                    temp = np.nanmean(results['results'][var].copy()[:,id_cs,id_state,:],axis=0).squeeze()
                idnan = [np.argwhere(np.isnan(temp[ics])).flatten() for ics in range(len(id_cs))]
                for ii,idn in enumerate(idnan):
                    if len(idn)==0:
                        idnan[ii] = temp.shape[1]
                    else:
                        idnan[ii] =idn[0]
                idend = np.asarray([np.max([np.min([idn-50,1000]),0]) for idn in idnan])
                temp_ = np.concatenate([temp[ics,idn:-1] for (ics,idn) in zip(range(len(id_cs)),idend)])
                temp_= temp_[~np.isnan(temp_)]
                temp_ = temp_[np.abs(temp_)<10]
                if len(temp_)>10:
                    cs_resp.append(temp_)
            if len(cs_resp) ==nb_cs:
                ll = np.min([len(ci) for ci in cs_resp])
                cs_resp_ = np.asarray([ci[:ll] for ci in cs_resp])
                cs_resp_n = (np.nanmean(cs_resp_,axis=1)-np.nanmean(cs_resp_[0,:]))/(np.nanmean(cs_resp_[2,:])-np.nanmean(cs_resp_[0,:]))
                cs_resp_n[cs_resp_n>1] = np.nan
                cs_resp_n[cs_resp_n<0] = np.nan
                cs_mu_n[ir,it,:] =cs_resp_n 
                cs_mu[ir,it,:] = [np.nanmean(_) for _ in cs_resp_]
    
    return cs_mu, cs_mu_n

def plot_var_baseline_iterations(agent,results_v,var, y_title,g_names,id_state,id_iter_plot,evtype = 'tau'):
    fig,ax = plt.subplots(figsize = (5,5))
    nb_cs = agent.task.nb_cs
    n_iterations_min = np.min([len(rr) for rr in results_v])
    base_resp_g=[]
    for (ir,results_group) in enumerate(results_v):
        base_resp = []
        for it in np.arange(n_iterations_min):
            id_tau = np.argmin( np.abs(results_group[it]['agent']['taus']-.5))
            results = results_group[id_iter_plot[it]]
            if evtype =='tau':
                temp =results['results'][var].copy()[id_tau,:,id_state,:].squeeze()
            elif evtype== 'mean':
                temp = np.nanmean(results['results'][var].copy()[:,:,id_state,:],axis=0).squeeze()
            idnan = [np.argwhere(np.isnan(temp[ics])).flatten() for ics in range(nb_cs)]
            for ii,idn in enumerate(idnan):
                if len(idn)==0:
                    idnan[ii] = temp.shape[1]
                else:
                    idnan[ii] =idn[0]
            idend = np.asarray([np.max([np.min([idn-50,1000]),0]) for idn in idnan])
            temp_ = np.concatenate([temp[ics,idn:-1] for (ics,idn) in zip(range(nb_cs),idend)])
            temp_ = temp_[~np.isnan(temp_)]
            base_resp.append(np.nanmean(temp_))
        base_resp_g.append(base_resp)
        ax.boxplot(base_resp_g)
        plot_config(ax,'',y_title,14,False)
        xticks_(ax,np.arange(len(g_names))+1,g_names)



    return base_resp_g
#%%