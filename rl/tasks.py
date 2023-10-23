import numpy as np

class TaskCueReward:
    def __init__(self, s): 

        self.probs = s['task']['prob'] 
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
        
        self.nb_cs = len(self.probs)
        self.trial_length = self.rew_onset + self.iti_dur+1

        self.deliv_rew =  np.concatenate(self.mags).flatten()
        self.deliv_cs =  np.concatenate(self.cs_id).flatten()
        self.n_deliv_rew = len(self.deliv_rew)

        self.create_csus_vectors()
        self.get_states_per_cs()

    def create_csus_vectors(self):
        mags  = self.mags
        probs = self.probs
        percent_per_cs = self.percent_per_cs
        n_trials = self.n_trials
        cs_vector =[]
        us_vector = [] 
        for ic in np.arange(self.nb_cs):
            rews = np.asarray(mags[ic])
            prs = np.asarray(probs[ic])
            n_tr_cue = np.int0(np.ceil(percent_per_cs[ic]*n_trials)[0])
            cs_vector.append(np.int0(ic*np.ones((1,n_tr_cue))).flatten())
            rv=[]
            for ir in np.arange(len(rews)):
                ntr_rw = np.int0( np.ceil(prs[ir]*n_tr_cue))
                rv.append(rews[ir]*np.ones( (ntr_rw,1)).flatten())
            rv = np.concatenate(rv)[0:n_tr_cue]
            us_vector.append(rv)
        us_vector = np.concatenate(us_vector) 
        cs_vector = np.concatenate(cs_vector)
        id_perm= np.arange(len(us_vector))
        np.random.shuffle(id_perm)
        us_vector = us_vector[id_perm]
        cs_vector = cs_vector[id_perm]

        self.us_vector = us_vector
        self.cs_vector = cs_vector

        return us_vector, cs_vector
    
    def get_states_per_cs(self):
        id_states_per_cs =[]
        n_types = self.nb_cs
        cue_onset = self.cue_onset
        rew_onset = self.rew_onset
        iti_dur = self.iti_dur
        for cs_id in np.arange(n_types):
            n_isi_states = len(np.arange(cue_onset,rew_onset+1))
            id_end = n_isi_states*(cs_id+1)+(cue_onset-1)
            id_st = id_end - n_isi_states+1
            id_end_last = n_isi_states*(n_types)+(cue_onset-1)
            last_states = np.arange(id_end_last+1,id_end_last+iti_dur+1)
            pre_states = np.arange(cue_onset)
            isi_states = np.arange(id_st,id_end+1)
            trial_vec = np.concatenate([pre_states,isi_states,last_states])
            id_states_per_cs.append(trial_vec)
        
        self.id_states_per_cs = id_states_per_cs

        return id_states_per_cs
            

    def sample_trial(self, n):
        reward  = self.us_vector[n]
        cs_id = self.cs_vector[n]
        rew_onset = self.rew_onset
        id_states =  self.id_states_per_cs[cs_id] 
        rew_trial = np.zeros((len(id_states)))
        rew_trial[rew_onset] = reward

        return id_states, rew_trial, reward,cs_id

