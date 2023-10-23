import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os 

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def title_(ax,title):
    ax.set_title(title)

def xticks_(ax,xticks,xticklabels=None):
    ax.set_xticks((xticks))
    if not xticklabels==None:
        ax.set_xticklabels((xticklabels))

def yticks_(ax,yticks,yticklabels=None):
    ax.set_yticks((yticks))
    if not yticklabels==None:
        ax.set_yticklabels((yticklabels))
def box_off(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def plot_config(ax,xlabel,ylabel,f_size,legend):
    ax.set_xlabel(xlabel,fontsize=f_size)
    ax.set_ylabel(ylabel,fontsize=f_size)
    box_off(ax)
    if legend is True:
        ax.legend(fontsize=f_size,framealpha=0)

def get_axlims(x, y, xlims, ylims):
    xlims[0] = np.nanmin((xlims[0], x.min()))
    xlims[1] = np.nanmax((xlims[1], x.max()))
    ylims[0] = np.nanmin((ylims[0], y.min()))
    ylims[1] = np.nanmax((ylims[1], y.max()))

    return xlims, ylims

def limit_xyaxis(ax):
    xlims = np.asarray([np.asarray(ia.get_xlim()).flatten() for  ia in ax.flatten()]).flatten()
    ylims = np.asarray([np.asarray(ia.get_ylim()).flatten() for  ia in ax.flatten()]).flatten()
    _ = [ia.set_xlim([np.min(xlims),np.max(xlims)])  for ia in ax.flatten()]
    _ = [ia.set_ylim([np.min(ylims),np.max(ylims)])  for ia in ax.flatten()]

    return xlims,ylims

def limit_xaxis(ax):
    xlims = np.asarray([np.asarray(ia.get_xlim()).flatten() for  ia in ax.flatten()]).flatten()
    _ = [ia.set_xlim([np.min(xlims),np.max(xlims)])  for ia in ax.flatten()]

    return xlims

def limit_yaxis(ax):
    ylims = np.asarray([np.asarray(ia.get_ylim()).flatten() for  ia in ax.flatten()]).flatten()
    _ = [ia.set_ylim([np.min(ylims),np.max(ylims)])  for ia in ax.flatten()]

    return ylims

def set_axlims(xlims, ylims, cf=None,exclude=[]):
    if cf is None:
        cf = plt.gcf()

    # expand limits slighty so that dots don't get clipped
    for lims in [xlims, ylims]:
        lims[0] -= .1 * (lims[1] - lims[0])
        lims[1] += .1 * (lims[1] - lims[0])

    for i_ax, ax in enumerate(cf.canvas.figure.get_axes()):
        if i_ax not in exclude:
            # set axis lims
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

def update_px(fig,tit,xtit,ytit,leg):

    fig.update_layout(
    xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Helvetica',
                size=12,
                color='rgb(82, 82, 82)',
                ),
            ),
    yaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor='rgb(204, 204, 204)',
            showline=True,
            showticklabels=True,
            ),
    autosize=True,
    showlegend=False,
    plot_bgcolor='white',
    title=tit,
    xaxis_title=xtit,
    yaxis_title=ytit,
    font_family="Helvetica"
    )   
    
# Plots for simulation results :
def plot_var_full_normalized_iterations(agent,results_v,var,title_var,y_title,cs_names,id_state,id_iter_plot,id_cs_plot):
    n_iterations = len(id_iter_plot)
    nb_cs = agent.task.nb_cs
    deliv_cs = agent.deliv_cs

    fig,ax =plt.subplots(1,2,figsize = (10,5))
    for results_group in results_v:
        cs_mu = np.zeros((n_iterations,nb_cs))
        cs_std = np.zeros((n_iterations,nb_cs))
        cs_mu_n = np.zeros((n_iterations,nb_cs))
        cs_std_n = np.zeros((n_iterations,nb_cs))
        for it in np.arange(len(id_iter_plot)):
            results = results_group[id_iter_plot[it]]
            cs_resp = []
            for ics in np.arange(nb_cs):
                id_cs = np.argwhere(deliv_cs==ics)
                temp = np.zeros_like(results[var])
                temp[:] = results[var]
                idnan = [np.argwhere(np.isnan(temp[1,ics,id_state,:].squeeze())).flatten()[0] for ics in id_cs]
                idnan = np.asarray([np.min([idn-50,1000]) for idn in idnan])
                idnan[idnan<0] = 0
                temp_ = np.concatenate([np.nanmean(temp[:,ics,id_state,idn:-1],axis=0).squeeze() for (ics,idn) in zip(id_cs,idnan)])
                cs_resp.append(temp_)
            ll = np.min([len(ci) for ci in cs_resp])
            cs_resp_ = np.asarray([ci[:ll] for ci in cs_resp])
            cs_resp_n = (cs_resp_-cs_resp_[0,:])/(cs_resp_[2,:]-cs_resp_[0,:])
            cs_resp_n[cs_resp_n>1] = np.nan
            cs_resp_n[cs_resp_n<0] = np.nan
            cs_mu[it,:] = [np.nanmean(_) for _ in cs_resp]
            cs_std[it,:] = [np.nanstd(_)/2 for _ in cs_resp]
            cs_mu_n[it,:] = np.nanmean(cs_resp_n,axis=1)
            cs_std_n[it,:] = np.nanstd(cs_resp_n,axis=1)/np.sqrt(cs_resp_n.shape[1]/2)
        ax[0].errorbar(id_cs_plot,np.nanmean(cs_mu[:,id_cs_plot],axis=0),np.nanstd(cs_mu[:,id_cs_plot],axis=0)/np.sqrt(n_iterations),color=cm.jet(it/n_iterations))
        ax[1].errorbar(id_cs_plot,np.nanmean(cs_mu_n[:,id_cs_plot],axis=0),np.nanstd(cs_mu_n[:,id_cs_plot],axis=0)/np.sqrt(n_iterations),color=cm.jet(it/n_iterations))
        plot_config(ax[0],'P(r)',y_title,14,False)
        plot_config(ax[1],'P(r)',y_title + '(normalized)',14,False)
        xticks_(ax[0],id_cs_plot,cs_names)
        xticks_(ax[1],id_cs_plot,cs_names)
        title_(ax[0],title_var)
        title_(ax[1],title_var + 'normalized')
    

    return fig, ax


# Plots for simulation results :
def plot_var_full_normalized(agent,results_v,var,title_var,y_title,cs_names,id_state,id_iter_plot,id_cs_plot):
    n_iterations = len(id_iter_plot)
    nb_cs = agent.task.nb_cs
    deliv_cs = agent.deliv_cs

    fig,ax =plt.subplots(1,2,figsize = (10,5))
    cs_mu = np.zeros((n_iterations,nb_cs))
    cs_std = np.zeros((n_iterations,nb_cs))
    cs_mu_n = np.zeros((n_iterations,nb_cs))
    cs_std_n = np.zeros((n_iterations,nb_cs))
    for it in np.arange(len(id_iter_plot)):
        results = results_v[id_iter_plot[it]]
        cs_resp = []
        for ics in np.arange(nb_cs):
            id_cs = np.argwhere(deliv_cs==ics)
            temp = np.zeros_like(results[var])
            temp[:] = results[var]
            idnan = [np.argwhere(np.isnan(temp[1,ics,id_state,:].squeeze())).flatten()[0] for ics in id_cs]
            idnan = np.asarray([np.min([idn-500,1000]) for idn in idnan])
            idnan[idnan<0] = 0
            temp_ = np.concatenate([np.nanmean(temp[:,ics,id_state,idn:-1],axis=0).squeeze() for (ics,idn) in zip(id_cs,idnan)])
            cs_resp.append(temp_)
        ll = np.min([len(ci) for ci in cs_resp])
        cs_resp_ = np.asarray([ci[:ll] for ci in cs_resp])
        cs_resp_n = (cs_resp_-cs_resp_[0,:])/(cs_resp_[2,:]-cs_resp_[0,:])
        cs_resp_n[cs_resp_n>1] = np.nan
        cs_resp_n[cs_resp_n<0] = np.nan
        cs_mu[it,:] = [np.nanmean(_) for _ in cs_resp]
        cs_std[it,:] = [np.nanstd(_)/2 for _ in cs_resp]
        cs_mu_n[it,:] = np.nanmean(cs_resp_n,axis=1)
        cs_std_n[it,:] = np.nanstd(cs_resp_n,axis=1)/np.sqrt(cs_resp_n.shape[1]/2)
        ax[0].errorbar(id_cs_plot,cs_mu[it,id_cs_plot],cs_std[it,id_cs_plot],color=cm.jet(it/n_iterations))
        ax[1].errorbar(id_cs_plot,cs_mu_n[it,id_cs_plot],cs_std_n[it,id_cs_plot],color=cm.jet(it/n_iterations))
    plot_config(ax[0],'P(r)',y_title,14,False)
    plot_config(ax[1],'P(r)',y_title + '(normalized)',14,False)
    xticks_(ax[0],id_cs_plot,cs_names)
    xticks_(ax[1],id_cs_plot,cs_names)
    title_(ax[0],title_var)
    title_(ax[1],title_var + 'normalized')
   

    return fig, ax

def plot_outcome_responses(results_v,var,id_state,id_tr_plot,id_iter_plot,y_title, title_var,cs_names):
    n_iterations = len(id_iter_plot)
    nb_cs = len(id_tr_plot)
    id_cs_plot = np.arange(nb_cs)
    fig,ax =plt.subplots(figsize = (5,5))
    cs_mu = np.zeros((n_iterations,nb_cs))
    cs_std = np.zeros((n_iterations,nb_cs))
    cs_num = np.zeros((n_iterations,nb_cs))
    cs_mu_n = np.zeros((n_iterations,nb_cs))
    for it in np.arange(len(id_iter_plot)):
        results = results_v[id_iter_plot[it]]
        cs_resp = []
        for ics in np.arange(len(id_tr_plot)):
            id_cs = id_tr_plot[ics]
            temp = np.zeros_like(results[var])
            temp[:] = results[var]
            temp_ = np.squeeze(np.nanmean(temp[:,id_cs,id_state,:],axis=1) )
            cs_resp.append(temp_)
        cs_mu[it,:] = [np.nanmean(_) for _ in cs_resp]
        cs_std[it,:] = [np.nanstd(_) for _ in cs_resp]
        cs_num[it,:] = [len(_) for _ in cs_resp]
        ax.errorbar(id_cs_plot,cs_mu[it,id_cs_plot],cs_std[it,id_cs_plot]/cs_num[it,id_cs_plot],color=cm.jet(it/n_iterations))
    plot_config(ax,'RPE',y_title,14,False)
    xticks_(ax,id_cs_plot,cs_names)
    title_(ax,title_var)

    return fig,ax


def plot_var_in_time(results_v,agent,var,id_iter_plot,y_title, title_var,cs_names,us_names,add_zero=False):
    id_states_per_cs = agent.id_states_per_cs
    n_iterations = len(id_iter_plot)
    nb_tr_types = len(agent.task.deliv_rew)
    deliv_rew = agent.task.deliv_rew
    deliv_cs = agent.task.deliv_cs

    xlims = np.nan*np.ones(2)
    ylims = np.nan*np.ones(2)
    fig,ax = plt.subplots(np.int0(nb_tr_types/2),2,figsize = (15,25))
    for it in np.arange(len(id_iter_plot)):
        results = results_v[id_iter_plot[it]]
        cs_vector = results['cs_vector']
        us_vector = results['us_vector']
        for ics in np.arange(nb_tr_types):
            temp = np.zeros_like(results[var])
            temp[:] = results[var]
            id_trials = np.argwhere((cs_vector==deliv_cs[ics]) * (us_vector==deliv_rew[ics]))
            id_states = id_states_per_cs[deliv_cs[ics]]
            temp_ = np.nanmean(temp[:,id_states,id_trials[-1]],axis=0)
            if add_zero is True:
                temp_[0] = 0
                temp_ = np.concatenate((temp_,np.asarray([0,])))
                time_v = np.arange(-1,len(temp_)-1)
            else:
                time_v = np.arange(-1,len(temp_)-1)
            
            us_id = np.remainder(ics,2)
            cs_id = deliv_cs[ics]
            ax[cs_id,us_id].plot(time_v,temp_,color=cm.jet(it/n_iterations))
            
            plot_config(ax[cs_id,us_id],'time re-cue',y_title,16,False)
            xlims, ylims = get_axlims(time_v, temp_, xlims, ylims)
            title_(ax[cs_id,us_id],cs_names[cs_id] + ' ' + us_names[us_id] + ' '+title_var )
    set_axlims(xlims, ylims)

    return fig, ax



def plot_distribution(results_v,agent,var,i_state,id_iter_plot,y_title, title_var,cs_names,us_names):
    id_states_per_cs = agent.id_states_per_cs
    n_iterations = len(id_iter_plot)
    nb_tr_types = len(agent.task.deliv_rew)
    deliv_rew = agent.task.deliv_rew
    deliv_cs = agent.task.deliv_cs
    taus = agent.taus
   
    id_sort = np.argsort(taus)
    xlims = np.nan*np.ones(2)
    ylims = np.nan*np.ones(2)
    fig,ax = plt.subplots(np.int0(nb_tr_types/2),2,figsize = (15,25))
    for it in np.arange(len(id_iter_plot)):
        results = results_v[id_iter_plot[it]]
        cs_vector = results['cs_vector']
        us_vector = results['us_vector']
        for ics in np.arange(nb_tr_types):
            temp = np.zeros_like(results[var])
            temp[:] = results[var]
            id_trials = np.argwhere((cs_vector==deliv_cs[ics]) * (us_vector==deliv_rew[ics]))
            id_states = id_states_per_cs[deliv_cs[ics]]
            temp_ = temp[:,id_states[i_state],id_trials[-1]]
            us_id = np.remainder(ics,2)
            cs_id = deliv_cs[ics]
            ax[cs_id,us_id].plot(taus[id_sort],temp_[id_sort],color=cm.jet(it/n_iterations))
            plot_config(ax[cs_id,us_id],r'$\tau$',y_title,16,False)
            xlims, ylims = get_axlims(taus, temp_, xlims, ylims)
            title_(ax[cs_id,us_id],cs_names[cs_id] + ' ' + us_names[us_id] + ' '+title_var )
    set_axlims(xlims, ylims)

    return fig, ax

def plot_distribution_cdf(results_v,agent,var,i_state,id_iter_plot,x_title, title_var,cs_names,us_names):
    id_states_per_cs = agent.id_states_per_cs
    n_iterations = len(id_iter_plot)
    nb_tr_types = len(agent.task.deliv_rew)
    deliv_rew = agent.task.deliv_rew
    deliv_cs = agent.task.deliv_cs
    taus = agent.taus
   
    id_sort = np.argsort(taus)
    xlims = np.nan*np.ones(2)
    ylims = np.nan*np.ones(2)
    fig,ax = plt.subplots(np.int0(nb_tr_types/2),2,figsize = (15,25))
    for it in np.arange(len(id_iter_plot)):
        results = results_v[id_iter_plot[it]]
        cs_vector = results['cs_vector']
        us_vector = results['us_vector']
        for ics in np.arange(nb_tr_types):
            temp = np.zeros_like(results[var])
            temp[:] = results[var]
            id_trials = np.argwhere((cs_vector==deliv_cs[ics]) * (us_vector==deliv_rew[ics]))
            id_states = id_states_per_cs[deliv_cs[ics]]
            temp_ = temp[:,id_states[i_state],id_trials[-1]]
            n_bins=100
            us_id = np.remainder(ics,2)
            cs_id = deliv_cs[ics]
            ax[cs_id,us_id].hist(temp_, n_bins, density=True, histtype='step',
                            cumulative=True, label='Empirical',color=cm.jet(it/n_iterations))
            plot_config(ax[cs_id,us_id],x_title,'cdf',16,False)
            xlims, ylims = get_axlims(temp_, temp_, xlims, ylims)
            title_(ax[cs_id,us_id],cs_names[cs_id] + ' ' + us_names[us_id] + ' '+title_var )
    # set_axlims(xlims, (0,1))

    return fig, ax


def plot_distribution_pdf(results_v,agent,var,i_state,id_iter_plot,x_title, title_var,cs_names,us_names,fig=None,ax=None):
    id_states_per_cs = agent.id_states_per_cs
    n_iterations = len(id_iter_plot)
    nb_tr_types = len(agent.task.deliv_rew)
    deliv_rew = agent.task.deliv_rew
    deliv_cs = agent.task.deliv_cs
    taus = agent.taus
   
    id_sort = np.argsort(taus)
    xlims = np.nan*np.ones(2)
    ylims = np.nan*np.ones(2)
    if fig is None:
        fig,ax = plt.subplots(np.int0(nb_tr_types/2),2,figsize = (15,25))
    for it in np.arange(len(id_iter_plot)):
        results = results_v[id_iter_plot[it]]
        cs_vector = results['cs_vector']
        us_vector = results['us_vector']
        for ics in np.arange(nb_tr_types):
            temp = np.zeros_like(results[var])
            temp[:] = results[var]
            id_trials = np.argwhere((cs_vector==deliv_cs[ics]) * (us_vector==deliv_rew[ics]))
            id_states = id_states_per_cs[deliv_cs[ics]]
            temp_ = temp[:,id_states[i_state],id_trials[-1]]
            n_bins=int( len(temp_/10))
            us_id = np.remainder(ics,2)
            cs_id = deliv_cs[ics]
            ax[cs_id,us_id].hist(temp_, n_bins, density=True, histtype='step',
                            cumulative=False, label='Empirical',color=cm.jet(it/n_iterations))
            plot_config(ax[cs_id,us_id],x_title,'cdf',16,False)
            xlims, ylims = get_axlims(temp_, temp_, xlims, ylims)
            title_(ax[cs_id,us_id],cs_names[cs_id] + ' ' + us_names[us_id] + ' '+title_var )
    # set_axlims(xlims, (0,1))

    return fig, ax


def plot_vars_convergence(results_v,taus_v,agent, vars,id_iter_plot,y_labels, plot_titles,cs_names,     \
                  simulation_type,fig_dir,save_tit):
    nb_cs = agent.task.nb_cs
    deliv_cs = agent.deliv_cs
    n_vars = len(vars)

    for it in np.arange(len(id_iter_plot)):
        results = results_v[id_iter_plot[it]]
        n_cells = len(taus_v)
        id_sort = np.argsort(taus_v)
        xlims = np.nan*np.ones(2)
        ylims = np.nan*np.ones(2)
        fig,ax =plt.subplots(nb_cs,n_vars,figsize = (15,15))
        for ivar in np.arange(n_vars):
            for ics in  np.arange(nb_cs):
                var = vars[ivar]
                temp = np.zeros_like(results[var])
                temp[:] = results[var]
                id_cs = np.argwhere(np.asarray(deliv_cs)==ics)
                temp_ = np.squeeze(np.nanmean(temp[:,id_cs,3,:],axis=1))
                for icell in np.arange(n_cells):
                    temp_cell = temp_[id_sort[icell],:]
                    temp_cell = temp_cell[~np.isnan(temp_cell)]
                    xv = np.arange(len(temp_cell))
                    ax[ics,ivar].plot(xv,temp_cell,color = cm.jet(icell/n_cells))
                    xlims, ylims = get_axlims(xv, temp_cell, xlims, ylims)
        
        for ivar in np.arange(n_vars):
            y_title =y_labels[ivar]
            title_var = plot_titles[ivar]
            for ics in  np.arange(nb_cs):
                title_(ax[ics,ivar],cs_names[ics]  + ' '+title_var )
                if ics== nb_cs-1:
                    plot_config(ax[ics,ivar],r'Trials',y_title,16,False)
                else:
                    plot_config(ax[ics,ivar],'',y_title,16,False)
                ax[ics,ivar].set_xlim(xlims)
    
    fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))

def plot_mean_per_trial_batch(agent,results_v,vars,id_states,save_titles,plot_titles,y_labels,cs_names,id_cs_plot,id_iter_plot,simulation_type,fig_dir):
    for iv in np.arange(len(vars)):
        var = vars[iv]
        id_state = id_states[iv]
        title_var = plot_titles[iv]
        y_title = y_labels[iv]
        save_tit = save_titles[iv]
        fig,ax = plot_var_full_normalized(agent,results_v,var,title_var,y_title,cs_names,id_state,id_iter_plot,id_cs_plot)
        fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))

def plot_mean_per_trial_batch_iterations(agent,results_v,vars,id_states,save_titles,plot_titles,y_labels,cs_names,id_cs_plot,id_iter_plot,simulation_type,fig_dir):
    for iv in np.arange(len(vars)):
        var = vars[iv]
        id_state = id_states[iv]
        title_var = plot_titles[iv]
        y_title = y_labels[iv]
        save_tit = save_titles[iv]
        fig,ax = plot_var_full_normalized_iterations(agent,results_v,var,title_var,y_title,cs_names,id_state,id_iter_plot,id_cs_plot)
        fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))



def plot_within_trial_batch(agent,results_v,vars,save_titles,plot_titles,y_labels,add_zeros,cs_names,us_names,id_iter_plot,simulation_type,fig_dir):
    for iv in np.arange(len(vars)):
        var = vars[iv]
        title_var = plot_titles[iv]
        y_title = y_labels[iv]
        save_tit = save_titles[iv]
        add_zero = add_zeros[iv]
        fig,ax = plot_var_in_time(results_v,agent,var,id_iter_plot,y_title, title_var,cs_names,us_names,add_zero)
        fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))




def plot_distributions_batch(agent,results_v,vars,save_titles,plot_titles,y_labels,i_states,cs_names,us_names,id_iter_plot,simulation_type,fig_dir):
    for iv in np.arange(len(vars)):
        var = vars[iv]
        title_var = plot_titles[iv]
        y_title = y_labels[iv]
        save_tit = save_titles[iv]
        i_state = i_states[iv]
        fig,ax = plot_distribution(results_v,agent,var,i_state,id_iter_plot,y_title, title_var,cs_names,us_names)

        fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))


def plot_distributions_cdf_batch(agent,results_v,vars,save_titles,plot_titles,x_labels,i_states,cs_names,us_names,id_iter_plot,simulation_type,fig_dir):
    for iv in np.arange(len(vars)):
        var = vars[iv]
        title_var = plot_titles[iv]
        x_title = x_labels[iv]
        save_tit = save_titles[iv]
        i_state = i_states[iv]
        fig,ax = plot_distribution_cdf(results_v,agent,var,i_state,id_iter_plot,x_title, title_var,cs_names,us_names)
        fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))


def plot_distributions_pdf_batch(agent,results_v,vars,save_titles,plot_titles,x_labels,i_states,cs_names,us_names,id_iter_plot,simulation_type,fig_dir):
    for iv in np.arange(len(vars)):
        var = vars[iv]
        title_var = plot_titles[iv]
        x_title = x_labels[iv]
        save_tit = save_titles[iv]
        i_state = i_states[iv]
        fig,ax = plot_distribution_pdf(results_v,agent,var,i_state,id_iter_plot,x_title, title_var,cs_names,us_names)
        fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))

def plot_distributions_pdf_overimposed(agent,results_v,vars,save_titles,plot_titles,x_labels,i_states,cs_names,us_names,id_iter_plot,simulation_type,fig_dir):
    for iv in np.arange(len(vars)):
        var = vars[iv]
        title_var = plot_titles[iv]
        x_title = x_labels[iv]
        save_tit = save_titles[iv]
        i_states_ = i_states[iv]
        nb_tr_types = len(agent.task.deliv_rew)
        
        for it in np.arange(len(id_iter_plot)):
            fig,ax = plt.subplots(np.int0(nb_tr_types/2),2,figsize = (15,25))
            for (iis,i_state) in enumerate(i_states_):
                id_states_per_cs = agent.id_states_per_cs
                n_iterations = len(id_iter_plot)
                
                deliv_rew = agent.task.deliv_rew
                deliv_cs = agent.task.deliv_cs
                taus = agent.taus
            
                id_sort = np.argsort(taus)
                xlims = np.nan*np.ones(2)
                ylims = np.nan*np.ones(2)
                    
                
                results = results_v[id_iter_plot[it]]
                cs_vector = results['cs_vector']
                us_vector = results['us_vector']
                for ics in np.arange(nb_tr_types):
                    temp = np.zeros_like(results[var])
                    temp[:] = results[var]
                    id_trials = np.argwhere((cs_vector==deliv_cs[ics]) * (us_vector==deliv_rew[ics]))
                    id_states = id_states_per_cs[deliv_cs[ics]]
                    temp_ = temp[:,id_states[i_state],id_trials[-1]]
                    n_bins=int( len(temp_/5))
                    us_id = np.remainder(ics,2)
                    cs_id = deliv_cs[ics]
                    ax[cs_id,us_id].hist(temp_, n_bins, density=True, histtype='step',
                                    cumulative=False, label='Empirical',color=cm.jet(iis/len(i_states_)))
                    plot_config(ax[cs_id,us_id],x_title,'cdf',16,False)
                    xlims, ylims = get_axlims(temp_, temp_, xlims, ylims)
                    title_(ax[cs_id,us_id],cs_names[cs_id] + ' ' + us_names[us_id] + ' '+title_var )
        # set_axlims(xlims, (0,1))

        # fig,ax = plot_distribution_pdf(results_v,agent,var,i_state,id_iter_plot,x_title, title_var,cs_names,us_names)
        # fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))


def limit_xaxis(ax):
    xlims = np.asarray([np.asarray(ia.get_xlim()).flatten() for  ia in ax.flatten()]).flatten()
    _ = [ia.set_xlim([np.min(xlims),np.max(xlims)])  for ia in ax.flatten()]

    return xlims

def limit_yaxis(ax):
    ylims = np.asarray([np.asarray(ia.get_ylim()).flatten() for  ia in ax.flatten()]).flatten()
    _ = [ia.set_ylim([np.min(ylims),np.max(ylims)])  for ia in ax.flatten()]

    return ylims


def plot_distributions_cdf_overimposed(agent,results_v,vars,save_titles,plot_titles,x_labels,i_states,cs_names,us_names,id_iter_plot,simulation_type,fig_dir):
    for iv in np.arange(len(vars)):
        var = vars[iv]
        title_var = plot_titles[iv]
        x_title = x_labels[iv]
        save_tit = save_titles[iv]
        i_states_ = i_states[iv]
        nb_tr_types = len(agent.task.deliv_rew)
        
        for it in np.arange(len(id_iter_plot)):
            fig,ax = plt.subplots(np.int0(nb_tr_types/2),2,figsize = (15,25))
            for (iis,i_state) in enumerate(i_states_):
                id_states_per_cs = agent.id_states_per_cs
                n_iterations = len(id_iter_plot)
                
                deliv_rew = agent.task.deliv_rew
                deliv_cs = agent.task.deliv_cs
                taus = agent.taus
            
                id_sort = np.argsort(taus)
                xlims = np.nan*np.ones(2)
                ylims = np.nan*np.ones(2)
                    
                
                results = results_v[id_iter_plot[it]]
                cs_vector = results['cs_vector']
                us_vector = results['us_vector']
                for ics in np.arange(nb_tr_types):
                    temp = np.zeros_like(results[var])
                    temp[:] = results[var]
                    id_trials = np.argwhere((cs_vector==deliv_cs[ics]) * (us_vector==deliv_rew[ics]))
                    id_states = id_states_per_cs[deliv_cs[ics]]
                    temp_ = temp[:,id_states[i_state],id_trials[-1]]
                    n_bins=100
                    us_id = np.remainder(ics,2)
                    cs_id = deliv_cs[ics]
                    ax[cs_id,us_id].hist(temp_, n_bins, density=True, histtype='step',
                                    cumulative=True, label='Empirical',color=cm.jet(iis/len(i_states_)))
                    plot_config(ax[cs_id,us_id],x_title,'cdf',16,False)
                    xlims, ylims = get_axlims(temp_, temp_, xlims, ylims)
                    title_(ax[cs_id,us_id],cs_names[cs_id] + ' ' + us_names[us_id] + ' '+title_var )
        # set_axlims(xlims, (0,1))

        # fig,ax = plot_distribution_pdf(results_v,agent,var,i_state,id_iter_plot,x_title, title_var,cs_names,us_names)
        # fig.savefig(os.path.join(fig_dir, save_tit + '_' + simulation_type + '.pdf'))

