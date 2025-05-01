import numpy as np
from scipy.optimize import root

from rl.tasks import *
import random

# Set random seed
seed_value = 1234  # Or any integer you like
np.random.seed(seed_value)
random.seed(seed_value)


#%%

def create_instance_agent_pav_task(taus_,eta_,epsilon_):
    s = dict()
    s['agent'] = dict()
    s['task'] = dict()
    s['agent']['taus'] =  np.sort(taus_)# np.asarray([.5,.5]) #
    s['agent']['eta'] = eta_
    s['agent']['str_agent'] = ['Eta_' + str(eta_)]#,'neutral','optimistic']
    s['agent']['epsilon'] = epsilon_
    s['agent']['base_alpha'] =1 
    s['agent']['gamma'] = 0.99
    s['agent']['do_sampling'] = False
    s['agent']['d1_d2'] = True
    s['task']['prob'] = [[.1, .9],[.5, .5],[.9, .1] ,[.8, .2]]
    s['task']['mags'] = [[1, 0],[1, 0],[1, 0] ,[-1, 0]]
    s['task']['percent_per_cs'] = 1/len(s['task']['mags'])*np.ones((len(s['task']['mags']),1))
    s['task']['cs_id'] = [[0,0],[1,1],[2,2],[3,3]]
    s['task']['str_cs'] = [ '10%','50%','90%','80%Puff']
    s['task']['str_task'] = 'drl_6odor'
    s['task']['n_trials'] = 30000
    s['task']['cue_onset'] = 1
    s['task']['cue_dur'] = 1
    s['task']['rew_onset'] = 3
    s['task']['iti_dur'] = 1
    s['task']['n_iterations'] = 1
    s['task']['type'] = 'cue_reward'
    s['task']['seed']=1234

    agent = TDLearning(s)

    return agent

class RWLearning:

    def __init__(self, s):
        
        self.seed = s['task']['seed']
        self.mags = s['task']['mags'] 
        self.cs_id = s['task']['cs_id']
        self.str_cs = s['task']['str_cs'] 
        self.str_task = s['task']['str_task']
        self.n_trials = s['task']['n_trials']
        self.cue_onset = s['task']['cue_onset'] 
        self.cue_dur = s['task']['cue_dur'] 
        self.rew_onset = s['task']['rew_onset']
        self.iti_dur = s['task']['iti_dur'] 
        self.percent_per_cs = s['task']['percent_per_cs']


        self.taus = s['agent']['taus']  
        self.str_agent = s['agent']['str_agent']  
        self.base_alpha = s['agent']['base_alpha'] 
        self.task_type = s['task']['type']
        n_cells = len(self.taus)

        if self.task_type =='cue_reward':
            self.task = TaskCueReward(s)

        if self.seed  is not None: 
            np.random.seed(self.seed)
            random.seed(self.seed)
        self.n_deliv_rew = self.task.n_deliv_rew
        self.deliv_rew = self.task.deliv_rew
        self.deliv_cs = self.task.deliv_cs
        self.trial_length = self.task.trial_length
        self.n_trials = self.task.n_trials
        self.cs_vector  = self.task.cs_vector
        self.us_vector  = self.task.us_vector
        self.n_types = len(self.task.mags)
        self.n_cells = len(self.taus)
        self.distribution = np.zeros((self.n_types,self.n_cells))   


        self.distribution_trials = np.nan*np.ones((self.n_types,self.n_cells,self.n_trials))
        self.deltas_trials= np.nan*np.ones((self.n_types,n_cells,self.n_trials))
        self.n_trials_per_type = np.zeros((self.n_types))

    def compute_td_error(self,r,vi_n,taus):
        delta           = r  - vi_n
        valence         = (delta >= 0.)
        sc_pos          = valence * taus
        sc_neg          = (1-valence) * (1-taus)
        resp = delta * (valence * taus+ (1. - valence) * (1-taus))

        return delta, valence, sc_pos, sc_neg,resp


    def update_values( self,sc_pos, sc_neg, delta,vi_n):
        pred_vals = vi_n+ (self.base_alpha * sc_pos* abs(delta)) - (self.base_alpha*sc_neg * abs(delta))

        return pred_vals
        
    def run_task(self):
        for n in np.arange(self.n_trials):
            # if np.remainder(n,1000)==0:
                # print('Trial '+ str(n) + ' of ' + str(self.n_trials)+  '  ||| ' )
            
            _, _, reward,cs_id = self.task.sample_trial(n)
            vi_n =  self.distribution[cs_id,:]
            delta, _, sc_pos, sc_neg ,resp = self.compute_td_error(reward,vi_n,self.taus)
            pred_vals = self.update_values( sc_pos, sc_neg, delta,vi_n)

            self.distribution[cs_id,:] = pred_vals
            self.distribution_trials[cs_id,:,int(self.n_trials_per_type[cs_id])] = pred_vals
            self.deltas_trials[cs_id,:,int(self.n_trials_per_type[cs_id])] = resp

            self.n_trials_per_type[cs_id] += 1

        return self.distribution,self.deltas_trials,self.distribution_trials, self.n_trials_per_type

    def reformat_results(self):
        self.results = {}
        self.results['distribution'] = self.distribution
        self.results['deltas_trials'] = self.deltas_trials
        self.results['distribution_trials'] = self.distribution_trials
        self.results['n_trials_per_type'] = self.n_trials_per_type

        return self.results
    

class TDLearning_avrew:

    def __init__(self, s):

        self.taus = s['agent']['taus']  
        self.str_agent = s['agent']['str_agent']  
        self.epsilon = s['agent']['epsilon']  
        self.base_alpha = s['agent']['base_alpha'] 
        self.base_step_hat = s['agent']['base_step_hat'] 
        self.random_init = s['agent']['random_init'] 
        self.gamma = s['agent']['gamma']  
        self.n_cells = len(self.taus)
        self.task_type = s['task']['type']
        self.do_shuffle = s['task']['do_shuffle']

        if self.task_type =='cue_reward':
            self.task = TaskCueReward(s,do_shuffle=self.do_shuffle)

        self.n_deliv_rew = self.task.n_deliv_rew
        
        self.deliv_rew = self.task.deliv_rew
        self.deliv_cs = self.task.deliv_cs
        
        self.trial_length = self.task.trial_length
        self.n_trials = self.task.n_trials
        self.cs_vector  = self.task.cs_vector
        self.us_vector  = self.task.us_vector
        self.id_states_per_cs = self.task.id_states_per_cs
        self.n_types = len(self.task.mags)

    def run_task(self):
        n_trials = self.n_trials
        for n in np.arange(n_trials):
            # if np.remainder(n,1000)==0:
                # print('Trial '+ str(n) + ' of ' + str(n_trials)+  '  ||| ' )
            
            id_states, rew_trial, reward,cs_id = self.task.sample_trial(n)
            self.run_trial(id_states,rew_trial)
            self.distribution_g[:,id_states] =  self.gs_new
            self.distribution_ng[:,id_states] = self.ngs_new
            self.distribution_g_hat =  self.gs_hat_new 
            self.distribution_ng_hat = self.ngs_hat_new 
            self.distribution[:,id_states]= self.vis_new 
            self.distribution_hat = self.vis_hat_new
            self.distribution_std[:,id_states]= self.vi_std_new
            self.deltas[:,id_states[1:]] = self.tds_new[:,0:-1]
            self.deltas_trials_raw[:,id_states]= self.tds_raw_new
            self.i_step_per_cs[cs_id] = self.i_step_per_cs[cs_id]+1

            self.update_results(id_states,n)

    def update_results(self,id_states,n):
        self.results['distribution_ng_hat_trials'][:,n] =self.ngs_hat_new 
        self.results['distribution_g_hat_trials'][:,n] = self.gs_hat_new
        self.results['distribution_hat_trials'][:,n]= self.vis_hat_new

        self.results['distribution_ng_trials'][:,id_states,n] =self.ngs_new
        self.results['distribution_g_trials'][:,id_states,n] = self.gs_new
        self.results['distribution_trials'][:,id_states,n]= self.vis_new
        self.results['deltas_trials'][:,id_states[1:],n] = self.tds_new[:,0:-1]
        
    def reformat_results(self):
        n_tr_max = self.n_tr_max
        deliv_rew = self.deliv_rew
        deliv_cs = self.deliv_cs
        cs_vector = self.cs_vector
        us_vector = self.us_vector
        n_cells = self.n_cells
        id_states_per_cs  = self.id_states_per_cs 
        dist_sink = ['distribution_trials_full' , 'distribution_g_trials_full' ,'distribution_ng_trials_full' , 'deltas_trials_full']
        dist_source = ['distribution_trials','distribution_g_trials','distribution_ng_trials','deltas_trials']
        for id in np.arange(len(dist_source)):
            dist_temp = self.results[dist_source[id]]
            for i_type in np.arange(len(deliv_rew)):
                i_cs = deliv_cs[i_type]
                id_states =  id_states_per_cs[i_cs]
                idCS = np.argwhere(cs_vector==i_cs)
                idUS = np.argwhere(us_vector==deliv_rew[i_type])
                id_trials = np.intersect1d(idCS,idUS)
                id_trials = id_trials[0:np.min([n_tr_max,len(id_trials)])]
                dist_ = dist_temp[:,id_states,:]
                dist_ = dist_[:,:,id_trials]
                self.results[dist_sink[id]][0:n_cells,i_type,:,0:dist_.shape[2]]=dist_
        dist_sink = ['distribution_trials_full2' , 'distribution_g_trials_full2' ,'distribution_ng_trials_full2' ]
        dist_source = ['distribution_trials','distribution_g_trials','distribution_ng_trials']
        ucs =np.unique(cs_vector)
        for id in np.arange(len(dist_source)):
            dist_temp = self.results[dist_source[id]]
            for i_cs in np.arange(len(ucs)):
                idcs = ucs[i_cs]
                id_states =  id_states_per_cs[i_cs]
                id_trials = np.argwhere(cs_vector==idcs).flatten()
                id_trials = id_trials[0:np.min([n_tr_max,len(id_trials)])]
                dist_ = dist_temp[:,id_states,:]
                dist_ = dist_[:,:,id_trials]
                print([dist_.shape,self.results[dist_sink[id]][0:n_cells,i_cs,:,0:dist_.shape[2]].shape,dist_temp.shape])
                self.results[dist_sink[id]][0:n_cells,i_cs,:,0:dist_.shape[2]]=dist_

    def run_trial(self,id_states,rew_trial):

        vis,gs,ngs,vis_hat,gs_hat,ngs_hat = self.get_current_values(id_states)
        self.initialize_trial(vis.shape)

        for iit in np.arange(len(id_states) -1):
            vi_n = vis[:,iit]
            vi_n_1 = vis[:,iit+1]

            go_vals = gs[:,iit]
            nogo_vals = ngs[:,iit]
            go_vals_hat = gs_hat
            nogo_vals_hat = ngs_hat

            r = rew_trial[iit]
            delta, valence, sc_pos, sc_neg = self.compute_td_error(r,vi_n_1,vi_n,vis_hat) 
            new_g,new_ng,new_g_hat,new_ng_hat,pred_vals,pred_val_std,pred_vals_hat,valence_upd = self.update_values( sc_pos, sc_neg, delta,go_vals, nogo_vals, go_vals_hat, nogo_vals_hat)
            #---------

            resp = delta * (valence * self.taus[:, np.newaxis] + (1. - valence) * (1-self.taus[:, np.newaxis]))
            mean_resp = np.mean(resp, axis=1)
            self.tds_raw_new[:,iit] = np.mean(delta ,axis=1)
            self.tds_new[:,iit] = mean_resp 

            self.gs_new[:,iit] = new_g
            self.ngs_new[:,iit] = new_ng

            self.gs_hat_new[:,iit] = new_g_hat 
            self.ngs_hat_new[:,iit] = new_ng_hat 
            
            self.vis_new[:,iit] = pred_vals 
            self.vis_hat_new[:,iit] = pred_vals_hat 
            self.vi_std_new[:,iit] = pred_val_std 
    

    def update_values(self, sc_pos, sc_neg, delta,go_vals, nogo_vals,go_vals_hat, nogo_vals_hat, do_pooling = True):

        mean_resp_neg = np.mean(sc_neg * abs(delta), axis=1)
        mean_resp_pos = np.mean(sc_pos * abs(delta), axis=1)

        update_g = self.eta * (mean_resp_pos) - self.epsilon * go_vals_hat
        update_n = (1-self.eta) * (mean_resp_neg )- self.epsilon * nogo_vals_hat

        update_g_hat = self.eta * (mean_resp_pos) - self.epsilon * go_vals_hat
        update_n_hat = (1-self.eta) * (mean_resp_neg )- self.epsilon * nogo_vals_hat


        new_g = go_vals + self.base_alpha*update_g
        new_ng = nogo_vals + self.base_alpha*update_n

        new_g_hat = go_vals_hat + self.base_step_hat*update_g_hat
        new_ng_hat = nogo_vals_hat + self.base_step_hat*update_n_hat

        pred_vals = (new_g - new_ng)
        pred_vals_hat = (new_g_hat - new_ng_hat)
        pred_val_std = (new_g + new_ng)
        valence_upd  = sc_pos + sc_neg

        return new_g,new_ng,new_g_hat,new_ng_hat,pred_vals,pred_val_std,pred_vals_hat,valence_upd

    def compute_td_error(self,r,samp_dist,vi_n,vi_hat_n):
        if len(samp_dist.shape) == 0:
            delta           = r + samp_dist - vi_n - vi_hat_n # TODO
            valence         = (delta >= 0.)
            sc_pos          = valence * self.taus
            sc_neg          = (1-valence) * (1-self.taus)
        else:
            delta           = r + samp_dist[np.newaxis, :] - vi_n[:, np.newaxis]- vi_hat_n[:, np.newaxis] # TODO
            valence         = (delta >= 0.)
            sc_neg          = (1.-valence) * (1-self.taus[:, np.newaxis])
            sc_pos          = (valence) * (self.taus[:, np.newaxis])

        return delta, valence, sc_pos, sc_neg


    def get_current_values(self, id_states):
        vis =  self.distribution[:,id_states]
        gs =  self.distribution_g[:,id_states]
        ngs = self.distribution_ng[:,id_states]

        vis_hat =  self.distribution_hat 
        gs_hat =  self.distribution_g_hat 
        ngs_hat = self.distribution_ng_hat 

        return vis,gs,ngs,vis_hat,gs_hat,ngs_hat

    def initialize_trial(self, tr_shape):

        self.vis_new = np.zeros((tr_shape)) 
        self.gs_new = np.zeros((tr_shape)) 
        self.ngs_new = np.zeros((tr_shape)) 

        self.vis_hat_new = np.zeros((tr_shape))  
        self.gs_hat_new = np.zeros((tr_shape)) 
        self.ngs_hat_new = np.zeros((tr_shape)) 

        self.vi_std_new = np.zeros((tr_shape)) 
        self.tds_new = np.zeros((tr_shape)) 
        self.tds_raw_new = np.zeros((tr_shape)) 

    def initialize_results(self):
        n_deliv_cs = self.n_deliv_cs
        random_init = self.random_init
        n_types = self.n_types
        n_cells = self.n_cells
        n_deliv_rew = self.n_deliv_rew
        trial_length  = self.trial_length
        n_trials = self.n_trials
        n_tr_max=int(np.ceil(n_trials/2))
        self.n_tr_max = n_tr_max
        if random_init:
            self.distribution = np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.distribution_hat = np.random.randn(n_cells,) #np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.distribution_std = np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.distribution_g = np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.distribution_ng = np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.distribution_g_hat = np.random.randn(n_cells, ) #np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.distribution_ng_hat = np.random.randn(n_cells,) #np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.samp_distribution = np.random.randn(n_cells, (trial_length-2)*n_types +2)
        else:
            self.distribution = np.zeros((n_cells, (trial_length-2)*n_types +2))
            self.distribution_hat = np.zeros((n_cells,) )
            self.distribution_std = np.zeros((n_cells, (trial_length-2)*n_types +2))
            self.distribution_g = np.zeros((n_cells, (trial_length-2)*n_types +2))
            self.distribution_ng = np.zeros((n_cells, (trial_length-2)*n_types +2))
            self.distribution_g_hat = np.zeros((n_cells, ) )
            self.distribution_ng_hat = np.zeros((n_cells,) )
            self.samp_distribution = np.zeros((n_cells, (trial_length-2)*n_types +2))

        self.deltas = np.zeros((n_cells,(trial_length-2)*n_types +2))
        self.deltas_trials_raw = np.zeros((n_cells,(trial_length-2)*n_types +2)) 
        self.i_step_per_cs  = [1, 1, 1, 1]

        results = dict()
        results['distribution_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))
        results['distribution_g_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew, trial_length,n_tr_max))
        results['distribution_ng_trials_full']= np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))
        results['distribution_hat_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew,n_tr_max)) #np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))
        results['distribution_g_hat_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew,n_tr_max)) #np.nan*np.ones((n_cells, n_deliv_rew, trial_length,n_tr_max))
        results['distribution_ng_hat_trials_full']= np.nan*np.ones((n_cells, n_deliv_rew,n_tr_max)) #np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))
        results['deltas_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))
        results['distribution_trials_full2'] = np.nan*np.ones((n_cells, n_deliv_cs,trial_length,n_tr_max))
        results['distribution_g_trials_full2'] = np.nan*np.ones((n_cells, n_deliv_cs, trial_length,n_tr_max))
        results['distribution_ng_trials_full2']= np.nan*np.ones((n_cells, n_deliv_cs,trial_length,n_tr_max))

        
        results['distribution_ng_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['distribution_g_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['distribution_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['distribution_ng_hat_trials'] = np.nan*np.ones((n_cells,n_trials)) #np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['distribution_g_hat_trials'] = np.nan*np.ones((n_cells,n_trials)) #np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['distribution_hat_trials'] = np.nan*np.ones((n_cells,n_trials)) #np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['deltas_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['update_gv'] = np.nan*np.ones((n_trials,n_cells))
        results['update_nv'] = np.nan*np.ones((n_trials,n_cells))
        results['grad_vi'] = np.nan*np.ones((n_trials,n_cells))

        self.results = results

class TDLearning:

    def __init__(self, s,random_init=None):

        self.taus = s['agent']['taus']  
        self.eta = s['agent']['eta']  
        self.str_agent = s['agent']['str_agent']  
        self.epsilon = s['agent']['epsilon']  
        self.base_alpha = s['agent']['base_alpha'] 
        self.do_sampling = s['agent']['do_sampling'] 
        self.gamma = s['agent']['gamma']  
        self.n_cells = len(self.taus)
        self.task_type = s['task']['type']
        self.seed = s['task']['seed']
        if self.seed  is not None: 
            np.random.seed(self.seed)
            random.seed(self.seed)

        if random_init is None:
            self.random_init=True
        else:
            self.random_init=random_init


        if self.task_type =='cue_reward':
            self.task = TaskCueReward(s)

        self.n_deliv_rew = self.task.n_deliv_rew
        self.deliv_rew = self.task.deliv_rew
        self.deliv_cs = self.task.deliv_cs
        self.n_deliv_cs =len(self.task.cs_id)
        self.trial_length = self.task.trial_length
        self.n_trials = self.task.n_trials
        self.cs_vector  = self.task.cs_vector
        self.us_vector  = self.task.us_vector
        self.id_states_per_cs = self.task.id_states_per_cs
        self.n_types = len(self.task.mags)

    def run_task(self):
        n_trials = self.n_trials
        for n in np.arange(n_trials):
            # if np.remainder(n,1000)==0:
                # print('Trial '+ str(n) + ' of ' + str(n_trials)+  '  ||| ' )
            
            id_states, rew_trial, reward,cs_id = self.task.sample_trial(n)
            self.run_trial(id_states,rew_trial)

            self.distribution_g [:,id_states] =  self.gs_new
            self.distribution_ng[:,id_states]= self.ngs_new
            self.distribution[:,id_states]= self.vis_new 
            self.distribution_std[:,id_states]= self.vi_std_new
            self.deltas[:,id_states[1:]] = self.tds_new[:,0:-1]
            self.deltas_trials_raw[:,id_states]= self.tds_raw_new
            self.samp_distribution[:,id_states] =  self.samp_new
            self.i_step_per_cs[cs_id] = self.i_step_per_cs[cs_id]+1

            self.update_results(id_states,n)

    def update_results(self,id_states,n):

        self.results['distribution_ng_trials'][:,id_states,n] =self.ngs_new
        self.results['distribution_g_trials'][:,id_states,n] = self.gs_new
        self.results['distribution_trials'][:,id_states,n]= self.vis_new
        self.results['deltas_trials'][:,id_states[1:],n] = self.tds_new[:,0:-1]
        
    def reformat_results(self):
        n_tr_max = self.n_tr_max
        deliv_rew = self.deliv_rew
        deliv_cs = self.deliv_cs
        cs_vector = self.cs_vector
        us_vector = self.us_vector
        n_cells = self.n_cells
        id_states_per_cs  = self.id_states_per_cs 
        dist_sink=['distribution_trials_full' , 'distribution_g_trials_full' ,'distribution_ng_trials_full' ,           'deltas_trials_full']
        dist_source=['distribution_trials','distribution_g_trials','distribution_ng_trials','deltas_trials']
        for id in np.arange(len(dist_source)):
            dist_temp = self.results[dist_source[id]]
            for i_type in np.arange(len(deliv_rew)):
                i_cs = deliv_cs[i_type]
                id_states =  id_states_per_cs[i_cs]
                idCS = np.argwhere(cs_vector==i_cs)
                idUS = np.argwhere(us_vector==deliv_rew[i_type])
                id_trials = np.intersect1d(idCS,idUS)
                id_trials =id_trials[(id_trials<dist_temp.shape[-1])]
                # id_trials = id_trials[0:np.min([n_tr_max,len(id_trials)])]
                dist_ = dist_temp[:,id_states,:]
                dist_ = dist_[:,:,id_trials]
                self.results[dist_sink[id]][0:n_cells,i_type,:,0:dist_.shape[2]]=dist_
        dist_sink = ['distribution_trials_full2' , 'distribution_g_trials_full2' ,'distribution_ng_trials_full2' ,  'deltas_trials_full2']
        dist_source = ['distribution_trials','distribution_g_trials','distribution_ng_trials','deltas_trials']
        ucs =np.unique(cs_vector)
        for id in np.arange(len(dist_source)):
            dist_temp = self.results[dist_source[id]]
            for i_cs in np.arange(len(ucs)):
                idcs = ucs[i_cs]
                id_states =  id_states_per_cs[i_cs]
                id_trials = np.argwhere(cs_vector==idcs).flatten()
                id_trials =id_trials[(id_trials<dist_temp.shape[-1])]
                # id_trials = id_trials[0:np.min([n_tr_max,len(id_trials)])]
                
                dist_ = dist_temp[:,id_states,:]
                dist_ = dist_[:,:,id_trials]
                # print([.shape,self.results[dist_sink[id]][0:n_cells,i_cs,:,0:dist_.shape[2]].shape,dist_temp.shape])
                self.results[dist_sink[id]][0:n_cells,i_cs,:,0:dist_.shape[2]]=dist_


    def run_trial(self,id_states,rew_trial):

        vis,gs,ngs = self.get_expectiles(id_states)
        self.initialize_trial(vis.shape)

        for iit in np.arange(len(id_states) -1):
            vi_n = vis[:,iit]
            vi_n_1 = vis[:,iit+1]
            go_vals = gs[:,iit]
            nogo_vals = ngs[:,iit]
            r = rew_trial[iit]
            if self.do_sampling is True:
                samp_dist = self.sample_distribution(vi_n_1)
            else:
                samp_dist = vi_n_1
            delta, valence, sc_pos, sc_neg = self.compute_td_error(r,samp_dist,vi_n)
            new_g,new_ng,pred_vals,pred_val_std,valence_upd = self.update_values( sc_pos, sc_neg, delta,go_vals, nogo_vals)

            #---------

            resp = delta * (valence * self.taus[:, np.newaxis] + (1. - valence) * (1-self.taus[:, np.newaxis]))
            mean_resp = np.mean(resp, axis=1)
            self.tds_raw_new[:,iit] = np.mean(delta ,axis=1)
            self.tds_new[:,iit] = mean_resp 
            
            self.gs_new[:,iit] = new_g
            self.ngs_new[:,iit] = new_ng
            self.vis_new[:,iit] = pred_vals 
            self.vi_std_new[:,iit] = pred_val_std 
            self.samp_new[:,iit] = samp_dist
    

    def update_values(self, sc_pos, sc_neg, delta,go_vals, nogo_vals, do_pooling = True):

        mean_resp_neg = np.mean(sc_neg * abs(delta), axis=1)
        mean_resp_pos = np.mean(sc_pos * abs(delta), axis=1)

        update_g = self.eta * (mean_resp_pos) - self.epsilon * go_vals
        update_n = (1-self.eta) * (mean_resp_neg )- self.epsilon * nogo_vals

        # update_g = self.eta*(sc_pos*abs(delta))-self.epsilon*go_vals
        # update_n = (1-self.eta)*(sc_neg*abs(delta))-self.epsilon*nogo_vals

        new_g = go_vals + self.base_alpha*update_g
        new_ng = nogo_vals + self.base_alpha*update_n
        pred_vals = (new_g - new_ng)
        pred_val_std = (new_g + new_ng)
        valence_upd  = sc_pos + sc_neg

        return new_g,new_ng,pred_vals,pred_val_std,valence_upd

    def compute_td_error(self,r,samp_dist,vi_n):
        if len(samp_dist.shape) == 0:
            delta           = r + self.gamma*samp_dist - vi_n
            valence         = (delta >= 0.)
            sc_pos          = valence * self.taus
            sc_neg          = (1-valence) * (1-self.taus)
        else:
            delta           = r + self.gamma*samp_dist[np.newaxis, :] - vi_n[:, np.newaxis]
            valence         = (delta >= 0.)
            sc_neg          = (1.-valence) * (1-self.taus[:, np.newaxis])
            sc_pos          = (valence) * (self.taus[:, np.newaxis])

        return delta, valence, sc_pos, sc_neg

    def expectile_root_fn(self,expectiles,taus, samples):
        # this function is zero when the samples come from a distribution consistent with the expectiles
        delta = samples[np.newaxis, :] - expectiles[:, np.newaxis]
        indic = np.array(delta <= 0., dtype=np.float32)
        grad = -2 * np.abs(taus[:, np.newaxis] - indic) * delta
        # take the mean over all samples

        return np.mean(grad, axis=1)

    def sample_distribution(self,expectiles):
        taus = self.taus
        fn_to_solve = lambda x: self.expectile_root_fn(expectiles,taus,x)
        samp_dist = root(fn_to_solve, x0=expectiles)['x']

        return samp_dist

    def get_expectiles(self, id_states):
        vis =  self.distribution[:,id_states]
        gs =  self.distribution_g[:,id_states]
        ngs = self.distribution_ng[:,id_states]

        return vis,gs,ngs

    def initialize_trial(self, tr_shape):

        self.vis_new = np.zeros((tr_shape)) 
        self.gs_new = np.zeros((tr_shape)) 
        self.ngs_new = np.zeros((tr_shape)) 
        self.vi_std_new = np.zeros((tr_shape)) 
        self.tds_new = np.zeros((tr_shape)) 
        self.tds_raw_new = np.zeros((tr_shape)) 
        self.samp_new = np.zeros((tr_shape)) 

    def initialize_results(self):
        n_types = self.n_types
        n_deliv_cs = self.n_deliv_cs
        n_cells = self.n_cells
        n_deliv_rew = self.n_deliv_rew
        random_init = self.random_init
        trial_length  = self.trial_length
        n_trials = self.n_trials
        n_tr_max= self.n_trials #int(np.ceil(n_trials/2))
        self.n_tr_max = n_tr_max

        results = dict()

        results['distribution_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))
        results['distribution_g_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew, trial_length,n_tr_max))
        results['distribution_ng_trials_full']= np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))
        results['deltas_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))
        results['distribution_trials_full2'] = np.nan*np.ones((n_cells, n_deliv_cs,trial_length,n_tr_max))
        results['deltas_trials_full2'] = np.nan*np.ones((n_cells, n_deliv_cs,trial_length,n_tr_max))
        
        results['distribution_g_trials_full2'] = np.nan*np.ones((n_cells, n_deliv_cs, trial_length,n_tr_max))
        results['distribution_ng_trials_full2']= np.nan*np.ones((n_cells, n_deliv_cs,trial_length,n_tr_max))
        
        if random_init:
            self.distribution = np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.distribution_std = np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.distribution_g = np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.distribution_ng = np.random.randn(n_cells, (trial_length-2)*n_types +2)
            self.samp_distribution = np.random.randn(n_cells, (trial_length-2)*n_types +2)
        else:
            self.distribution = np.zeros((n_cells, (trial_length-2)*n_types +2))
            self.distribution_std = np.zeros((n_cells, (trial_length-2)*n_types +2))
            self.distribution_g = np.zeros((n_cells, (trial_length-2)*n_types +2))
            self.distribution_ng = np.zeros((n_cells, (trial_length-2)*n_types +2))
            self.samp_distribution = np.zeros((n_cells, (trial_length-2)*n_types +2))


        # self.distribution = np.random.randn(n_cells, (trial_length-2)*n_types +2)
        # self.distribution_std = np.random.randn(n_cells, (trial_length-2)*n_types +2)
        # self.distribution_g = np.random.randn(n_cells, (trial_length-2)*n_types +2)
        # self.distribution_ng = np.random.randn(n_cells, (trial_length-2)*n_types +2)
        # self.samp_distribution = np.random.randn(n_cells, (trial_length-2)*n_types +2)
        self.deltas = np.zeros((n_cells,(trial_length-2)*n_types +2))
        self.deltas_trials_raw = np.zeros((n_cells,(trial_length-2)*n_types +2)) 
        self.i_step_per_cs  = np.ones(self.n_deliv_cs)
        
        results['distribution_ng_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['distribution_g_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['distribution_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['deltas_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))

        results['update_gv'] = np.nan*np.ones((n_trials,n_cells))
        results['update_nv'] = np.nan*np.ones((n_trials,n_cells))
        results['grad_vi'] = np.nan*np.ones((n_trials,n_cells))

        self.results = results

class TDLearning_V:

    def __init__(self, s):

        self.taus = s['agent']['taus']  
        self.eta = s['agent']['eta']  
        self.str_agent = s['agent']['str_agent']  
        self.epsilon = s['agent']['epsilon']  
        self.base_alpha = s['agent']['base_alpha'] 
        self.do_sampling = s['agent']['do_sampling'] 
        self.gamma = s['agent']['gamma']  
        self.n_cells = len(self.taus)
        self.task_type = s['task']['type']
        self.seed = s['task']['seed']
        if self.seed  is not None: 
            np.random.seed(self.seed)
            random.seed(self.seed)


        if self.task_type =='cue_reward':
            self.task = TaskCueReward(s)

        self.n_deliv_rew = self.task.n_deliv_rew
        self.deliv_rew = self.task.deliv_rew
        self.deliv_cs = self.task.deliv_cs
        self.trial_length = self.task.trial_length
        self.n_trials = self.task.n_trials
        self.cs_vector  = self.task.cs_vector
        self.us_vector  = self.task.us_vector
        self.id_states_per_cs = self.task.id_states_per_cs
        self.n_types = len(self.task.mags)

    def run_task(self):
        n_trials = self.n_trials
        for n in np.arange(n_trials):
            # if np.remainder(n,1000)==0:
                # print('Trial '+ str(n) + ' of ' + str(n_trials)+  '  ||| ' )
            
            id_states, rew_trial, reward,cs_id = self.task.sample_trial(n)
            self.run_trial(id_states,rew_trial)

            self.distribution[:,id_states]= self.vis_new 
            self.deltas[:,id_states[1:]] = self.tds_new[:,0:-1]
            self.deltas_trials_raw[:,id_states]= self.tds_raw_new
            self.samp_distribution[:,id_states] =  self.samp_new
            self.i_step_per_cs[cs_id] = self.i_step_per_cs[cs_id]+1

            self.update_results(id_states,n)

    def update_results(self,id_states,n):

        self.results['distribution_trials'][:,id_states,n]= self.vis_new
        self.results['deltas_trials'][:,id_states[1:],n] = self.tds_new[:,0:-1]
        
    def reformat_results(self):
        n_tr_max = self.n_tr_max
        deliv_rew = self.deliv_rew
        deliv_cs = self.deliv_cs
        cs_vector = self.cs_vector
        us_vector = self.us_vector
        n_cells = self.n_cells
        id_states_per_cs  = self.id_states_per_cs 
        dist_sink=['distribution_trials_full' , 'deltas_trials_full']
        dist_source=['distribution_trials','deltas_trials']
        for id in np.arange(len(dist_source)):
            dist_temp = self.results[dist_source[id]]
            for i_type in np.arange(len(np.unique(cs_vector))):
                i_cs = deliv_cs[i_type]
                id_states =  id_states_per_cs[i_cs]
                idCS = np.argwhere(cs_vector==i_cs)
                idUS = np.argwhere(us_vector==deliv_rew[i_type])
                id_trials = np.intersect1d(idCS,idUS)
                id_trials = id_trials[0:np.min([n_tr_max,len(id_trials)])]
                dist_ = dist_temp[:,id_states,:]
                dist_ = dist_[:,:,id_trials]
                self.results[dist_sink[id]][0:n_cells,i_type,:,0:dist_.shape[2]]=dist_

    def run_trial(self,id_states,rew_trial):

        vis = self.get_expectiles(id_states)
        self.initialize_trial(vis.shape)

        for iit in np.arange(len(id_states) -1):
            vi_n = vis[:,iit]
            vi_n_1 = vis[:,iit+1]
            r = rew_trial[iit]
            if self.do_sampling is True:
                samp_dist = self.sample_distribution(vi_n_1)
            else:
                samp_dist = vi_n_1
            delta, valence, sc_pos, sc_neg = self.compute_td_error(r,samp_dist,vi_n)
            pred_vals ,valence_upd = self.update_values( sc_pos, sc_neg, delta,vi_n)

            resp = delta * (valence * self.taus[:, np.newaxis] + (1. - valence) * (1-self.taus[:, np.newaxis]))
            mean_resp = np.mean(resp, axis=1)
            self.tds_raw_new[:,iit] = np.mean(delta ,axis=1)
            self.tds_new[:,iit] = mean_resp 
            
            self.vis_new[:,iit] = pred_vals 
            self.samp_new[:,iit] = samp_dist
    

    def update_values(self, sc_pos, sc_neg, delta,vi_n,do_pooling=True):
        if do_pooling is True and self.do_sampling is False:
            mean_resp_neg = np.mean(sc_neg * abs(delta), axis=1)
            mean_resp_pos = np.mean(sc_pos * abs(delta), axis=1)
        else:   
            mean_resp_neg = sc_neg * abs(delta)
            mean_resp_pos = sc_pos * abs(delta)

        update_pos = self.eta * (mean_resp_pos)  
        update_neg = (1-self.eta) * (mean_resp_neg ) 
        valence_upd  = sc_pos + sc_neg

        pred_vals = vi_n+ (self.base_alpha*update_pos) - (self.base_alpha*update_neg)

        return pred_vals,valence_upd

    def compute_td_error(self,r,samp_dist,vi_n):
        if len(samp_dist.shape) == 0:
            delta           = r + self.gamma*samp_dist - vi_n
            valence         = (delta >= 0.)
            sc_pos          = valence * self.taus
            sc_neg          = (1-valence) * (1-self.taus)
        else:
            delta           = r + self.gamma*samp_dist[np.newaxis, :] - vi_n[:, np.newaxis]
            valence         = (delta >= 0.)
            sc_neg          = (1.-valence) * (1-self.taus[:, np.newaxis])
            sc_pos          = (valence) * (self.taus[:, np.newaxis])

        return delta, valence, sc_pos, sc_neg

    def expectile_root_fn(self,expectiles,taus, samples):
        # this function is zero when the samples come from a distribution consistent with the expectiles
        delta = samples[np.newaxis, :] - expectiles[:, np.newaxis]
        indic = np.array(delta <= 0., dtype=np.float32)
        grad = -2 * np.abs(taus[:, np.newaxis] - indic) * delta
        # take the mean over all samples

        return np.mean(grad, axis=1)

    def sample_distribution(self,expectiles):
        taus = self.taus
        fn_to_solve = lambda x: self.expectile_root_fn(expectiles,taus,x)
        samp_dist = root(fn_to_solve, x0=expectiles)['x']

        return samp_dist

    def get_expectiles(self, id_states):
        vis =  self.distribution[:,id_states]
        
        return vis

    def initialize_trial(self, tr_shape):

        self.vis_new = np.zeros((tr_shape)) 
        self.tds_new = np.zeros((tr_shape)) 
        self.tds_raw_new = np.zeros((tr_shape)) 
        self.samp_new = np.zeros((tr_shape)) 

    def initialize_results(self):
        n_types = self.n_types
        n_cells = self.n_cells
        n_deliv_rew = self.n_deliv_rew
        trial_length  = self.trial_length
        n_trials = self.n_trials
        n_tr_max=int(np.ceil(n_trials/2))
        self.n_tr_max = n_tr_max

        results = dict()

        results['distribution_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))
        results['deltas_trials_full'] = np.nan*np.ones((n_cells, n_deliv_rew,trial_length,n_tr_max))

        self.distribution = np.random.randn(n_cells, (trial_length-2)*n_types +2)
        self.samp_distribution = np.random.randn(n_cells, (trial_length-2)*n_types +2)
        self.deltas = np.zeros((n_cells,(trial_length-2)*n_types +2))
        self.deltas_trials_raw = np.zeros((n_cells,(trial_length-2)*n_types +2)) 
        self.i_step_per_cs  = [1, 1, 1, 1]
        
        results['distribution_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))
        results['deltas_trials'] = np.nan*np.ones((n_cells, (trial_length-2)*n_types +2,n_trials))

        results['grad_vi'] = np.nan*np.ones((n_trials,n_cells))

        self.results = results

