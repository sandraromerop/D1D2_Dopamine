
#%%


import glob
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.model_fitting import sigmoid,sigmoid_fun,slope_log,fit_sigmoid
from utils.plots import *

root_dir_data = '/Users/sromeropinto/Library/CloudStorage/GoogleDrive-sromeropinto@g.harvard.edu/My Drive/Data/LHb_Analysis/'
savedir =os.path.join(root_dir_data,'Analysis','BiophysicalModel','drug_experiments')
file_list = glob.glob(savedir + '/*formatteda.matbromocriptine_results_only_base.npy')

#%% Load single unit simulations 
res_mats = []
for id_unit in tqdm(range(len(file_list))):
    res = np.load(file_list[id_unit],allow_pickle=True).item()
    res_mats.append(res)

#%%
drug_concv = 10**(np.linspace(-1.5,2,10)) 
efficacies = np.linspace(.1,.9,6)
ndrug = len(drug_concv)
nunits = len(res_mats)
neffs = len(efficacies)
nda = res['da'].shape[1]
idbase = np.arange(1000,2000)
#%% Fit sigmoid to the curves for d2 activation: 

__,axo = plt.subplots(neffs,neffs,figsize=(neffs*3,neffs*3))
__,axa = plt.subplots(neffs,neffs,figsize=(neffs*3,neffs*3))

ec50_act, bias_act = np.zeros((neffs,neffs,ndrug)), np.zeros((neffs,neffs,ndrug))
ec50_occ, bias_occ = np.zeros((neffs,neffs,ndrug)), np.zeros((neffs,neffs,ndrug))
popts_occ,popts_act = np.zeros((neffs,neffs,ndrug,4)), np.zeros((neffs,neffs,ndrug,4))

for i_eff_d2l in range(neffs):
    for i_eff_d2s in range(neffs):
        xfit = np.linspace(-1,4.5,20)
        yy = sigmoid_fun(10**xfit,10)
        axo[i_eff_d2l,i_eff_d2s].plot(xfit,yy,color='grey')
        axa[i_eff_d2l,i_eff_d2s].plot(xfit,yy,color='grey')
        for idr,drug_conc in zip(range(ndrug),drug_concv):
            temp_act = np.nan*np.ones((nda,len(file_list)))
            temp_occ = np.nan*np.ones((nda,len(file_list)))
            temp_da = np.nan*np.ones((nda,len(file_list)))
            for id_unit in range(len(res_mats)):
                temp_da[:,id_unit] = np.nanmean(res_mats[id_unit]['da'][0,:,i_eff_d2l,i_eff_d2s,idbase],axis=0)
                temp_act[:,id_unit] = np.nanmean( res_mats[id_unit]['d2l_act'][idr,:,i_eff_d2l,i_eff_d2s,idbase],axis=0)
                temp_occ[:,id_unit] = np.nanmean(np.sum(res_mats[id_unit]['d2l_occ'][idr,:,i_eff_d2l,i_eff_d2s,idbase],axis=2),axis=0)   
            mu_act = np.nanmean(temp_act,axis=1) 
            mu_occ = np.nanmean(temp_occ,axis=1) 
            mu_da = np.nanmean(temp_da,axis=1) 
            
            popt_occ, _,yfit_occ = fit_sigmoid(mu_da,mu_occ,xfit,log_xscale=True)
            popt_act, _,yfit_act = fit_sigmoid(mu_da,mu_act,xfit,log_xscale=True)

            axo[i_eff_d2l,i_eff_d2s].plot(np.log10(mu_da),mu_occ,'o',color=cm.jet(idr/len(drug_concv)) )
            axo[i_eff_d2l,i_eff_d2s].plot(xfit,yfit_occ,color=cm.jet(idr/len(drug_concv)) )
            axa[i_eff_d2l,i_eff_d2s].plot(np.log10(mu_da),mu_act,'o',color=cm.jet(idr/len(drug_concv)) )
            axa[i_eff_d2l,i_eff_d2s].plot(xfit,yfit_act,color=cm.jet(idr/len(drug_concv)) )
            ec50_act[i_eff_d2l,i_eff_d2s,idr] = popt_act[1]
            bias_act[i_eff_d2l,i_eff_d2s,idr] = popt_act[-1]
            popts_act[i_eff_d2l,i_eff_d2s,idr,:] = popt_act
            bias_occ[i_eff_d2l,i_eff_d2s,idr] = popt_occ[-1]
            ec50_occ[i_eff_d2l,i_eff_d2s,idr] = popt_occ[1]
            popts_occ[i_eff_d2l,i_eff_d2s,idr,:] = popt_occ

        plot_config(axo[i_eff_d2l,i_eff_d2s],'log[DA] nM','D2l occ',14,False)
        plot_config(axa[i_eff_d2l,i_eff_d2s],'log[DA] nM','D2l act',14,False)
#%%
i_eff_d2l,i_eff_d2s= -1,-1


log_delta = np.log10(1/3)
ec50_d1 = 1000
ec50_d2 = 10
xfit = np.linspace(1,2.5,20)
n_points = len(xfit)
d1_eta, d2_eta_act,eta_act,d2_eta_occ,eta_occ  = [],[],[],[],[]

for idrug in np.arange(-1,ndrug):
    d1temp,d2temp_act,d2temp_occ =[],[],[]
    if idrug>-1:
        popt_act = popts_act[i_eff_d2l,i_eff_d2s,idrug,:]
        popt_occ = popts_occ[i_eff_d2l,i_eff_d2s,idrug,:]
    for id  in np.arange(n_points):
        if idrug>-1:
            slope_log_d2_act = slope_log(xfit[id],log_delta,popt_act,type_sigm ='full')
            slope_log_d2_occ = slope_log(xfit[id],log_delta,popt_occ,type_sigm ='full')
        else:
            slope_log_d2_act = slope_log(10**xfit[id],1/3,ec50_d2,type_sigm ='simple')
            slope_log_d2_occ = slope_log(10**xfit[id],1/3,ec50_d2,type_sigm ='simple')
        slope_log_d1 = slope_log(10**xfit[id],3,ec50_d1,type_sigm ='simple')
        d2temp_act.append(slope_log_d2_act)
        d1temp.append(slope_log_d1)
        d2temp_occ.append(slope_log_d2_occ)
        
    d1temp = np.asarray(d1temp)
    d2temp_act = np.asarray(d2temp_act)
    d2temp_occ = np.asarray(d2temp_occ)
    d1_eta.append(d1temp)
    d2_eta_act.append(d2temp_act)
    eta_act.append(np.divide(d1temp,(d1temp+d2temp_act)))
    d2_eta_occ.append(d2temp_occ)
    eta_occ.append(np.divide(d1temp,(d1temp+d2temp_occ)))

drug_concp = np.concatenate([np.asarray([0]),drug_concv])
fig,ax = plt.subplots(1,3,figsize=(18,7))
for idrug in np.arange(ndrug+1):
    ax[0].plot(xfit,eta_act[idrug],color=cm.viridis(idrug/ndrug),
        label=str(np.ceil(drug_concp[idrug]*100)/100)+ ' nM')
    ax[1].plot(xfit,2*eta_act[idrug]-1,color=cm.viridis(idrug/ndrug))
    diff_ = (2*eta_act[idrug]-1)-(2*eta_act[0]-1)
    ax[2].plot(xfit,diff_,color=cm.viridis(idrug/ndrug))
    plot_config(ax[0],'log[DA] nM',r'$\tau$',14,True)
    plot_config(ax[1],'log[DA] nM',r'2$\tau$-1',14,False)
    plot_config(ax[2],'log[DA] nM',r'$\Delta($2$\tau$-1)',14,False)
# plt.savefig(os.path.join(figdir,'_pop_' +  choose_drug + '_taus_from_d1d2_activation.pdf'))

fig,ax = plt.subplots(1,3,figsize=(18,7))
for idrug in np.arange(ndrug+1):
    ax[0].plot(xfit,eta_occ[idrug],color=cm.viridis(idrug/ndrug),
        label=str(np.ceil(drug_concp[idrug]*100)/100)+ ' nM')
    ax[1].plot(xfit,2*eta_occ[idrug]-1,color=cm.viridis(idrug/ndrug))
    diff_ = (2*eta_occ[idrug]-1)-(2*eta_occ[0]-1)
    ax[2].plot(xfit,diff_,color=cm.viridis(idrug/ndrug))    
    plot_config(ax[0],'log[DA] nM',r'$\tau$',14,True)
    plot_config(ax[1],'log[DA] nM',r'2$\tau$-1',14,False)
    plot_config(ax[2],'log[DA] nM',r'$\Delta($2$\tau$-1)',14,False)


#%% Now test for parameter sensitivity
