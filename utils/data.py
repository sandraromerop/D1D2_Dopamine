#%%
import os
from tqdm import tqdm
import glob
import pickle
import scipy.io as scio
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gsmooth
import mat73
#%%

def timesToIndices(times, startEnd, resolution):
    # given a time window [startEnd(1) startEnd(2)], sampled at "resolution"
    # converts the range [times(1) times(2)] to a list of indices in that time window
    assert(times(1) >= startEnd(1))
    assert(times(2) <= startEnd(2))
    indices = np.arange(1 + (times[0] - startEnd[0])/resolution,(times[1] - startEnd[0]) / resolution)

    return indices

def computeSmoothingKernel(smoothingTimeConst, psthResolution):
    # function [smLength,smoothingKernel] = computeSmoothingKernel(smoothingTimeConst, psthResolution)
    smoothingFunc = lambda t: np.dot((1 - np.exp(-t)),np.exp(-t/smoothingTimeConst))
    # smoothingFunc = @(t) (1 - exp(-t)).*exp(-t/smoothingTimeConst); 
    tv = np.arange(0,(2.5*smoothingTimeConst),step=psthResolution)
    smoothingKernel = smoothingFunc(tv)
    smoothingKernel = smoothingKernel / sum(smoothingKernel)
    smLength = len(smoothingKernel)

    return smLength,smoothingKernel

def get_trial_types(events):
    nameTypes = [ '90%R', '90%OR','50%R', '50%OR', '10%R', '10%OR', '80%R',  '80%OP', 'freeR', 'freeP', 'noR']
    # Initialize mats
    trialTypeId = np.zeros((len(events.odorOn),1)) 
    idPerTrial = dict()
    nPerTrialType = np.zeros((12,))
    allIds =[]

    # features: ids, nb pf trials, percent trials, odor id, oiginal trialtype
    # Odor on/off : free/cued
    # Rwd on/off:  rewarded/unrewarded
    # Odor type
    # Puff on/off

    features = np.zeros((len(events.odorOn),4))  
    for i in range(len(events.odorOn)):  
        mat = [events.odorOn[i] , events.odorID[i], events.rewardOn[i], events.airpuffOn[i]]
        features[i,:] = mat
    
    uO = np.unique( events.odorID) 
    uO = uO[~np.isnan(uO)] 
    puffID  = np.argwhere(np.isnan(features[:,3])==0)     # all trials with puff
    puffOdorOn = features[puffID,0]             # odor onset of all trials with puff+odor/no odor
    puffOdorId = puffID 
    puffOdorId = puffOdorId[~np.isnan(puffOdorOn)]             
    puff80P = puffOdorId 
    puffTrueOdor = np.unique(features[puffOdorId,1])  # odor id type 
    puffTrueOdor = puffTrueOdor[~np.isnan(puffTrueOdor)]   # odor id type 
    #% %
    # No odor + puff
    freeP = np.intersect1d(puffID, np.argwhere(np.isnan(features[:,1])==1)) 
    # freeP = intersect(puffID, find(isnan(features(:,2))==1)) 
    keepv = [] 
    for i in np.arange(len(puffTrueOdor)):
        odorondisc = features[np.argwhere(features[:,1]==puffTrueOdor[i]),3] 
        sumodordisc= sum(~np.isnan(odorondisc)) 
        if sumodordisc>5:
            keepv = np.concatenate((keepv,[i]))
    puffTrueOdor = puffTrueOdor[np.int16(keepv)] 
    # Odor + no Puf
    trialsOdorPuff = np.argwhere((features[:,1]==puffTrueOdor)) 
    trialsOdorPuff = trialsOdorPuff[np.isnan(features[trialsOdorPuff,3])]
    puff80O = trialsOdorPuff 
    #% %
    # All rewarded trials 
    rewID = np.argwhere((np.isnan(features[:,2])!=1)) 
    rewOdorId = np.unique(features[rewID,1]) 
    # Odors for rewarded trials 
    rewOdorId = rewOdorId[~np.isnan(rewOdorId)] 
    # Find if there's an unrewarded odor (will be cosidered 10# CS for some cells)
    takenOdors = np.concatenate((puffTrueOdor, rewOdorId))  
    allOdors = np.unique(features[~np.isnan(features[:,1]),1])
    xy, x_ind, id2 = np.intersect1d(takenOdors,allOdors, return_indices=True)
    allOdors[id2] =np.NaN 
    allOdors = allOdors[~np.isnan(allOdors)]
    #% %
    # Make sure some of the trials with 
    if len(allOdors)>0:
        for ii in np.arange(len(allOdors)):
            isRewarded =  np.unique(features[(features[:,1]==allOdors[ii]),2]) 
            isRewarded = isRewarded[~np.isnan(isRewarded)]
            isPuffed =  np.unique(features[(features[:,1]==allOdors[ii]),3]) 
            isPuffed = isPuffed[~np.isnan(isRewarded)]
            if len(isRewarded)==0 and len(isRewarded)==0:
                rewOdorId = np.concatenate((allOdors, rewOdorId))  
    # No odor + reward 
    freeR = np.intersect1d(rewID, np.argwhere(np.isnan(features[:,0])==1))
    #% %
    # Odor + reward/not reward
    percs = [.90,.50,.10]
    idPerc = np.zeros((3,2))
    idPerc[0,:] = np.int16([0,1])
    idPerc[1,:] = np.int16([2,3])  
    idPerc[2,:] = np.int16([4,5]) 
    names = [['90R','90O'],['50R','50O'],['10R','10O']]
    allCSId = []  
    allIds = [] 
    percRew = np.zeros((len(rewOdorId,)))
    for i in np.arange(len(rewOdorId)):
        # Trials for this odor
        id = np.argwhere(features[:,1]== rewOdorId[i]) 
        rwdLogic = features[id,2] 
        percRew[i]  = sum(~np.isnan(rwdLogic))/len(rwdLogic) 
        rewR = id[~np.isnan(rwdLogic)] 
        rewO = id[np.isnan(rwdLogic)] 
        idType =  np.argmin(abs(percs-percRew[i]))
        trialTypeId[rewR,0 ] = idPerc[idType,0] 
        trialTypeId[rewO,0 ] = idPerc[idType,1] 
        idPerTrial[names[idType][0]] = rewR 
        idPerTrial[names[idType][1]] = rewO 
        print
        nPerTrialType[np.int16(idPerc[idType,0])] = len(rewR) 
        nPerTrialType[np.int16(idPerc[idType,1])] = len(rewO) 
        allCSId = np.concatenate((allCSId , rewR,rewO))
    allIds = np.concatenate((allIds , allCSId))
    allIds = np.concatenate((allIds , puff80P,puff80O,freeP))  
    allPuffId = np.concatenate((puff80P , puff80O,freeP))
    # With odor and no rwd        
    noR = np.intersect1d( np.argwhere(np.isnan(features[:,2])==1),
                np.argwhere(np.isnan(features[:,1])!=1))
    # Discard the ones belonging to other CS
    [a, b, c]=np.intersect1d(noR,allCSId, return_indices=True)
    noR = np.delete(noR,b)
    [a, b,c ]=np.intersect1d(noR,allPuffId, return_indices=True)
    noR = np.delete(noR,b)
    allIds = np.concatenate((allIds , noR)) 
    #% % Fill in all mats
    trialTypeId[puff80P,0] = 7 
    trialTypeId[puff80O,0] = 8 
    trialTypeId[freeP,0] = 10 
    trialTypeId[freeR,0] = 9 
    trialTypeId[noR,0] = 11   
    missingTrials=np.argwhere(trialTypeId[:,0]==0) 
    idPerTrial['puffP'] = puff80P 
    idPerTrial['puffO'] = puff80O 
    idPerTrial['freeP'] = freeP  
    idPerTrial['freeR'] = freeR 
    idPerTrial['noR'] = noR  
    idPerTrial['missingTrials'] = missingTrials 
    nPerTrialType[6] = len(puff80P) 
    nPerTrialType[7] = len(puff80O) 
    nPerTrialType[9] = len(freeP) 
    nPerTrialType[8] = len(freeR) 
    nPerTrialType[10] = len(noR) 
    nPerTrialType[11] = len(missingTrials) 
    repeated = len(allIds)>len(events.odorOn) 
    
    return trialTypeId , idPerTrial, nPerTrialType, nameTypes, repeated


def spikes_to_rate( dt, spikes, synch = 'auto', adjust_t = False, tmax = None):
    """Reading spike-timestamps into a firing rate vector. 
    
    
    :param dt: timestep of firing rate vector
    :type dt: float
    :param spikes: Timestamps of action potentials
    :type spikes: numpy array
    :param synch: FWHM in s for other neurons in ensemble. If set to 'auto', the synch is decided based on average firing rate of the input cell. Default is 'auto'
    :type synch: float or 'auto'
    :param adjust_t: adjust offset of time series? Use this if *spikes* has a gap at the beginning.  Default is *False*
    :type adjust_t: bool
    :param tmax: Length of simulation in seconds, default None. If *tmax* is None, tmax will be equal to the last spike in the recording. 
    :type tmax: float
    
    """
    
    
    nspikes = spikes.size 
    DT = spikes[-1] - spikes[0]
    mNU = (nspikes - 1)/DT 
    mISI = 1/mNU
    
    if synch == 'auto':
        # print("Using automatic smoothing\n")

        synch = 0.3012*mISI  #smoothing in magic window
        W = synch/dt 
    else:
        W = synch/dt 
    
    if adjust_t:
        DTtrans = spikes[0]
        # print("Moving spikes by" , DTtrans , ' s')
        spikes += - DTtrans  
    
    if tmax is None:
        tmax = spikes[-1] 
    
    binedge = np.arange(0, tmax, dt) 
    tfile = binedge[:-1] + 0.5*dt
    sphist = np.histogram(spikes, binedge)[0]
    
    NUfile = gsmooth(sphist.astype(float), W,mode = 'wrap')/dt 
    
    lastspike = spikes[-1] + mISI 
    endindx = tfile > lastspike 
    NUfile[endindx] = mNU 
    
    # SANDRA: pad starting with mNU as well         
    n_sec_base = .1 # sec of MNU
    # DT = spikes[20] - spikes[0]
    # mNUb = (10 - 1)/DT

    n_samp_base = np.divide(n_sec_base,dt).astype(int)
    NUfile[0:n_samp_base] = mNU

    return spikes, NUfile, mNU, synch

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    try:
        data = scio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    except:
        data = mat73.loadmat(filename)

    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def get_data_dir(nas_drive, protocol, mouse, type_dir, type_data):


    if type_dir == 'extracted_su':
        dir_ = os.path.join(nas_drive, 'Sandra','Extracted_Data', protocol, type_data, mouse, 'SingleUnitData')
    elif type_dir == 'extracted_session' and type_data=='Photometry':
        dir_ = os.path.join(nas_drive, 'Sandra','Extracted_Data', protocol, type_data, mouse, 'SessionData')
    elif type_dir == 'extracted_session' and type_data=='Bpod':
        dir_ = os.path.join(nas_drive, 'Sandra','Extracted_Data', protocol, type_data, mouse )
    elif type_dir == 'extracted_session' and type_data=='Bonsai':
        dir_ = os.path.join(nas_drive, 'Sandra','Extracted_Data', protocol, type_data, mouse )
    elif type_dir == 'raw' and type_data=='Bonsai':
        dir_ = os.path.join(nas_drive, 'Sandra','Data', protocol, type_data, mouse, 'Data')
    elif type_dir == 'raw' and type_data=='Bpod':
        dir_ = os.path.join(nas_drive, 'Sandra','Data', protocol, type_data, mouse,protocol, 'SessionData')

    return dir_

# def read_cvs_row(full_name, irow):
#     with open(full_name,'r') as file:
#             reader = csv.reader(file)
#             gp = []
#             for row in reader:
#                 if len(row)>0:
#                     gp.append(row[irow])
            
#     return gp

def load_pickle(fname):
    file = open(fname,'rb')
    object_file = pickle.load(file)
    file.close()

    return object_file

def get_cell_range(path_to_data):
    
    file_list = glob.glob(os.path.join(os.path.join(path_to_data,'spikes'),"*.json"))
    cell_ids = np.unique([ii.split(os.sep)[-1].split('.json')[0].split('_')[-1] for ii in file_list])
    cell_ids = [int(c) for c in cell_ids]
    cell_range = range(min(cell_ids),max(cell_ids))

    return cell_range

def bin_spikes_list(spikes ,win_range,ms_per_bin):
    bin_len = np.int0((win_range[-1]-win_range[0])/ms_per_bin)
    time_bin =  np.linspace(win_range[0],win_range[-1],bin_len+1)
    spikes_binned = {}
    for cell in range(len(spikes)):
        num_trials = len(spikes[cell])
        spikes_binned[cell] = np.zeros((int(num_trials), bin_len))
        for trial_index, event_times in enumerate(spikes[cell][:num_trials]):
            sp_new, bin_edges = np.histogram(event_times,bins=time_bin,density=False)
            spikes_binned[cell][trial_index,:len(sp_new)] = sp_new
    time_bin  = time_bin[:bin_len]
    spbins = [spikes_binned[ii] for ii in range(len(spikes_binned))]
    sp_ = np.asarray([[spbins[icell][ii] for ii in np.arange(len(spbins[icell]))] for icell in range(len(spbins))])

    return spikes_binned,time_bin,sp_

def bin_spikes(spikes ,win_range,ms_per_bin):
    bin_len = np.int0((win_range[-1]-win_range[0])/ms_per_bin)
    time_bin =  np.linspace(win_range[0],win_range[-1],bin_len+1)
    spikes_binned = {}
    for cell in spikes:
        num_trials = len(spikes[cell])
        spikes_binned[cell] = np.zeros((int(num_trials), bin_len))
        for trial_index, event_times in enumerate(spikes[cell][:num_trials]):
            sp_new, bin_edges = np.histogram(event_times,bins=time_bin,density=False)
            spikes_binned[cell][trial_index,:len(sp_new)] = sp_new
    time_bin  = time_bin[:bin_len]

    return spikes_binned,time_bin


def get_mice_dates_list(path_data,suffs):
    u_dates_ =[]
    id_unique_ = []
    mice_v = []
    mice_all = []
    dates_all = []
    for i_suff in np.arange(len(suffs)):
        suffix = suffs[i_suff]
        data_path = os.path.join(path_data,suffix)
        ff = glob.glob(data_path + '/*formatted.mat')
        dates_,mice_ = [],[]
        for iff in np.arange(len(ff)):
            f_name = (ff[iff].split('/')[-1])
            id_start = f_name.find('_',2)+1
            fn_st = f_name[id_start:]
            id_end  = fn_st.find('_')
            dates_.append(fn_st[:id_end])
            id_start = f_name.find('_',1)+1
            fn_st = f_name[id_start:]
            id_end  = fn_st.find('_')
            mice_.append(fn_st[:id_end])
        
        mice_ = np.asarray(mice_)
        u_dates,id_unique =np.unique(dates_, return_index=True)
        u_dates_.append(u_dates)
        id_unique_.append(id_unique)
        mice_v.append(mice_[id_unique])
        mice_all.append(mice_)
        dates_all.append(dates_)
    
    return mice_v,mice_all,dates_all, id_unique_,u_dates_

def load_lick_raster(licks,odor_on,trial_types,tr_win,psth_resolution,psth_length):
    icount = 0
    n_trials  = len(trial_types)
    lick_raster = np.zeros((n_trials, psth_length+1))
    
    itrial_ids = []
    for i_trial in np.arange(n_trials):
        if trial_types[i_trial] != 10 and trial_types[i_trial] != 9 and  trial_types[i_trial] != 7  and  trial_types[i_trial] != 8:
            st_time = odor_on[i_trial]
            win_indices = (licks > st_time + tr_win[0])*(licks < st_time + tr_win[1])
            lick_times = licks[win_indices]
            lick_psth_indices = np.intp( 1 + np.round((lick_times - st_time) / psth_resolution) - tr_win[0] / psth_resolution)
            lick_r = np.zeros((psth_length+1))
            for i_lick in np.arange(len(lick_psth_indices)):
                lick_r[lick_psth_indices] += 1
            lick_raster[icount,:] = lick_r
            icount +=1
            itrial_ids.append(i_trial)
    ids_ = np.asarray(itrial_ids)
    lick_mu = np.nanmean(lick_raster,axis=1)

    return lick_mu, ids_

def map_cues(map_,trial_types):
    cue_types = np.zeros((len(trial_types)))
    for ic in range(len(map_)):
        ids = np.argwhere(trial_types==ic+1)
        cue_types[ids] = map_[ic]

    return cue_types

def map_rewards(rew_on, puff_on,trial_types):
    rew_types = np.zeros((len(trial_types,)))
    rew_types[~np.isnan(rew_on)] = 1
    rew_types[~np.isnan(puff_on)] = -1

    return rew_types


def replace_zeros_to_nan(var):
    var[var==0]=np.nan
    
    return var



def digitize_responses(lead_var, dependent_var,bins_v):
    n_bins = len(bins_v)
    da_xx = np.histogram_bin_edges(lead_var, bins=bins_v)
    ids_bins = np.digitize(lead_var ,da_xx)
    lead_var_bin, dependent_var_bin=[],[]
    for ib in np.arange(n_bins):
        ids_=np.argwhere(ids_bins==ib)
        if len(ids_)>=2:
            lead_var_bin.append(np.nanmean(lead_var[ids_]))
            dependent_var_bin.append(np.nanmean(dependent_var[ids_])) 
        else:
            lead_var_bin.append(np.nan)
            dependent_var_bin.append(np.nan) 

    return lead_var_bin, dependent_var_bin


#%%

# Classes
class DataLoad:
    def __init__(self, s):
        self.baselineWindow      = s['baselineWindow'] #[-1000 0];  % in units of milliseconds
        self.psthWindow          = s['psthWindow'] #[-1500 4500];  % in units of milliseconds
        self.cueWindow           = s['cueWindow'] #[0 400];  % in units of milliseconds
        self.traceWindow         = s['traceWindow'] #[500 2000];  % in units of milliseconds
        self.outcomeWindow       = s['outcomeWindow'] #[2000 2600];  % in units of milliseconds
        self.outcomeWindowLate       = s['outcomeWindowLate'] #[2200 2600];  % in units of milliseconds
        self.outcomeWindowFreeLate   = s['outcomeWindowFreeLate'] #[200 400];  % in units of milliseconds
        self.psthResolution          = s['psthResolution'] #2;  % in units of milliseconds
        self.normalizeTo50           = s['normalizeTo50'] #true; % when true, this will normalize the 50% responses for each cell according to the mean (across cells) of 50% responses in that animal.
        self.smoothingTimeConst      = s['smoothingTimeConst'] #20;  % 20 is what the eshel paper used (units of milliseconds)
        self.interestingTrialTypes   = s['interestingTrialTypes'] #1:15
        self.maxTrialsPerType        = s['maxTrialsPerType']
        self.indeces_puff80     = s['indeces']['puff80'] 
        self.indeces_prob10     = s['indeces']['prob10']
        self.indeces_prob50     = s['indeces']['prob50']
        self.indeces_prob90     = s['indeces']['prob90']
        self.indeces_freeRew    = s['indeces']['freeRew'] 
        self.indeces_freePuff   = s['indeces']['freePuff']

        self.psthLength          = np.int0(1 + np.ceil((self.psthWindow[1] - self.psthWindow[0]) / self.psthResolution)) # number of bins in the PSTH)
        self.psthTimes           = np.linspace(self.psthWindow[0], self.psthWindow[1], self.psthLength)   # relative time of each bin in the PSTH
        self.cuePeriod           =  np.int0(self.timesToIndices(self.cueWindow, self.psthWindow)) #immediately after cue
        self.tracePeriod       =  np.int0(self.timesToIndices(self.traceWindow, self.psthWindow ) )
        self.outcomePeriod       =  np.int0(self.timesToIndices(self.outcomeWindow, self.psthWindow ))
        self.outcomePeriodFreeLate  =  np.int0(self.timesToIndices(self.outcomeWindowFreeLate, self.psthWindow  ))
        self.outcomePeriodLate      =  np.int0(self.timesToIndices(self.outcomeWindowLate, self.psthWindow ))
        self.baselinePeriod = np.int0(self.timesToIndices(self.baselineWindow, self.psthWindow))
        self.nTrialTypes         = len(self.interestingTrialTypes)
        
        self.computeSmoothingKernel()
        

    def computeSmoothingKernel(self):
        smoothingFunc = lambda t: (1 - np.exp(-t))*np.exp(-t/self.smoothingTimeConst)
        # smoothingFunc = @(t) (1 - exp(-t)).*exp(-t/smoothingTimeConst); 
        tv = np.arange(0,(2.5*self.smoothingTimeConst),step=self.psthResolution)
        smoothingKernel = smoothingFunc(tv)
        smoothingKernel = np.divide(smoothingKernel, np.sum(smoothingKernel))
        smLength = len(smoothingKernel)
        self.smLength = smLength
        self.smoothingKernel  = smoothingKernel

        return smLength,smoothingKernel

    def timesToIndices(self,times, startEnd):
        # given a time window [startEnd(1) startEnd(2)], sampled at "resolution"
        # converts the range [times(1) times(2)] to a list of indices in that time window
        
        assert(times[0] >= startEnd[0])
        assert(times[1] <= startEnd[1])
        indices = np.arange(1 + (times[0] - startEnd[0])/self.psthResolution,(times[1] - startEnd[0]) / self.psthResolution)

        return indices
    
    def get_cells_same_day(self,file_list):
        # Get cells from same day : 
        dateF,tte,dateNumF = [],[],[]
        for filedId in file_list:
            date_str = filedId.split('_')[4]
            date_int = int(date_str.replace('-',''))
            dateF.append(filedId[4])
            dateNumF.append(int(date_str.replace('-','')))
        uF,idU = np.unique(dateNumF,return_index=True)
        self.nDates = len(np.unique(dateNumF))
        self.uF = uF
        self.idU = idU
    
    def get_num_mice(self,file_list):
        miceName = ['']
        # Get numb of mice: 
        for filedId in file_list:
            filedId = filedId.split('_')[3]
            if len(np.intersect1d(miceName,filedId))==0:
                miceName.append(filedId)
        miceName = miceName[1:]
        self.nMice = len(miceName)
        self.nUnits = len(file_list) 

    def compute_lick_responses(self,PSTH_licks):
        self.cueResponses_licks = np.asarray([[[np.nanmean(PSTH_licks[icell,itype,itrial,self.cuePeriod]) for itrial  in range(PSTH_licks.shape[2])] for itype  in range(PSTH_licks.shape[1])] for icell in range(PSTH_licks.shape[0])] )
        self.outcomeResponses_licks = np.asarray([[[np.nanmean(PSTH_licks[icell,itype,itrial,self.outcomePeriod]) for itrial  in range(PSTH_licks.shape[2])] for itype  in range(PSTH_licks.shape[1])] for icell in range(PSTH_licks.shape[0])] )
        self.traceResponses_licks = np.asarray([[[np.nanmean(PSTH_licks[icell,itype,itrial,self.tracePeriod]) for itrial  in range(PSTH_licks.shape[2])] for itype  in range(PSTH_licks.shape[1])] for icell in range(PSTH_licks.shape[0])] )

        return self.cueResponses_licks,self.outcomeResponses_licks,self.traceResponses_licks

    def compute_responses_re_baseline(self,PSTH):
        self.baselines = np.asarray([np.nanmean(PSTH[icell,:,:,self.baselinePeriod]) for icell in range(PSTH.shape[0])])
        self.PSTHs = PSTH - self.baselines[:,np.newaxis,np.newaxis,np.newaxis]
        self.cueResponses = np.asarray([[[np.nanmean(self.PSTHs[icell,itype,itrial,self.cuePeriod]) for itrial  in range(self.PSTHs.shape[2])] for itype  in range(self.PSTHs.shape[1])] for icell in range(self.PSTHs.shape[0])] )
        self.outcomeResponses = np.asarray([[[np.nanmean(self.PSTHs[icell,itype,itrial,self.outcomePeriod]) for itrial  in range(self.PSTHs.shape[2])] for itype  in range(self.PSTHs.shape[1])] for icell in range(self.PSTHs.shape[0])] )
        self.traceResponses = np.asarray([[[np.nanmean(self.PSTHs[icell,itype,itrial,self.tracePeriod]) for itrial  in range(self.PSTHs.shape[2])] for itype  in range(self.PSTHs.shape[1])] for icell in range(self.PSTHs.shape[0])] )

        return self.baselines,self.cueResponses,self.outcomeResponses,self.traceResponses, self.PSTHs

    def compute_mu_responses_per_cue(self, responses,idU = None):
        if idU is None:
            idU = np.arange(responses.shape[0])
        responses = responses[idU,:,:]
        indeces_resp = [self.indeces_prob10,self.indeces_prob50,self.indeces_prob90]
        mu_across_cells = np.asarray([np.nanmean(np.nanmean(responses[:,ii,:],axis=(1,2)),axis=0) for ii in indeces_resp])
        sem_across_cells = np.asarray([np.nanstd(np.nanmean(responses[:,ii,:],axis=(1,2)),axis=0)/np.sqrt(responses.shape[0]) for ii in indeces_resp])
        per_cell_mean = np.asarray([ np.nanmean(responses[:,ii,:],axis=(1,2)) for ii in indeces_resp])
        per_cell = np.asarray([ np.nanmean(responses[:,ii,:],axis=(1,2)) for ii in indeces_resp])

        return mu_across_cells,sem_across_cells,per_cell

    def compute_mu_responses_per_cue_outcome(self, responses,idU = None):
        if idU is None:
            idU = np.arange(responses.shape[0])
        responses = responses[idU,:,:]
        indeces_resp = [self.indeces_prob90[1],self.indeces_prob50[1],self.indeces_prob10[1],
                        self.indeces_prob90[0],self.indeces_prob50[0],self.indeces_prob10[0]]

        mu_across_cells = np.asarray([np.nanmean(np.nanmean(responses[:,ii,:],axis=1),axis=0) 
                                    for ii in indeces_resp])

        sem_across_cells = np.asarray([np.nanstd(np.nanmean(responses[:,ii,:],axis=1),axis=0)/np.sqrt(responses.shape[0]) 
                                    for ii in indeces_resp])
        per_cell = np.asarray([ np.nanmean(responses[:,ii,:],axis=1) 
                                    for ii in indeces_resp])

        return mu_across_cells,sem_across_cells,per_cell
    
    def load_cells(self,file_list, response_type = 'spikes'):
        nCells  = len(file_list)
        self.nCells = nCells
        PSTC_           = np.nan*np.ones((nCells,1, self.maxTrialsPerType*3, self.psthLength))  #  peri-stimulus time counts, including all cells
        PSTC            = np.nan*np.ones((nCells, self.nTrialTypes, self.maxTrialsPerType, self.psthLength) ) #  peri-stimulus time counts, including all cells
        spikeRaster     = np.nan*np.ones((nCells, self.nTrialTypes, self.maxTrialsPerType, self.psthLength))
        for iCell in tqdm(range(nCells),'Cell Number'):
            with open(file_list[iCell], 'rb') as handle:
                unit = pickle.load(handle)
            iTrialOfType = np.zeros((self.nTrialTypes,))   # keep track (separately per cell) of how many trials of each type have happened
            nTrials = len(unit['data']['events']['odorOn'])
            trialTypes = unit['data']['TrialTypes']
            
            for iTrial in  range(nTrials):
                trialTypeId  = trialTypes[iTrial]
                if trialTypes[iTrial]!=10 and trialTypes[iTrial]!=9 :
                    stTime = unit['data']['events']['odorOn'][iTrial];  # an odor-on event
                else:
                    stTime = np.asarray([unit['data']['events']['rewardOn'][iTrial], unit['data']['events']['airpuffOn'][iTrial]])   # an odor-on event
                    stTime = stTime[np.logical_not(np.isnan(stTime))]
                    stTime = np.min(stTime)
                # % Take a temporal window around the odor onset time
                # % Find the spikes that happened in that window, and map their times to PSTH indices
                if response_type == 'spikes':
                    windowIndices = np.logical_and(unit['data']['responses']['spike'] >= stTime + self.psthWindow[0] , unit['data']['responses']['spike'] <= stTime + self.psthWindow[1])
                    spikeTimes = unit['data']['responses']['spike'][windowIndices]
                else:
                    windowIndices = np.logical_and(unit['data']['responses']['lick'] >= stTime + self.psthWindow[0] , unit['data']['responses']['lick'] <= stTime + self.psthWindow[1])
                    spikeTimes = unit['data']['responses']['lick'][windowIndices]
                spikePsthIndices = np.round( np.divide(spikeTimes - stTime, self.psthResolution) ) - self.psthWindow[0] / self.psthResolution
                spikePsthIndices = [np.int0(ii) for ii in spikePsthIndices]
                trialTypeId  = trialTypes[iTrial]
                # % Increment the PSTH with these spikes
                trialTypeIx = trialTypeId-1
                if trialTypeIx >= 0 and trialTypeIx <= self.nTrialTypes and iTrialOfType[trialTypeIx]<=self.maxTrialsPerType:
                    PSTC[iCell, trialTypeIx, np.int0(iTrialOfType[trialTypeIx]), :] = np.zeros((1,1,1,self.psthLength))
                    PSTC_[iCell, 0,iTrial, :] = np.zeros((1,1,self.psthLength))
                    spikeRaster[iCell, trialTypeIx, np.int0(iTrialOfType[trialTypeIx]), :] = np.zeros((1,1,1,self.psthLength))
                    for iSpike in range(len(spikePsthIndices)):
                        tbs = np.int0(spikePsthIndices[iSpike] )
                        tbe = np.int0(min(tbs+self.smLength, self.psthLength))
                        PSTC[iCell, trialTypeIx, np.int0(iTrialOfType[trialTypeIx]), tbs:tbe] = PSTC[iCell, trialTypeIx, np.int0(iTrialOfType[trialTypeIx]), tbs:tbe] + self.smoothingKernel[:(tbe-tbs)] # smooth each spike as we count it
                        PSTC_[iCell,0, iTrial, tbs:tbe] = PSTC_[iCell, 0,iTrial, tbs:tbe] +self.smoothingKernel[:(tbe-tbs)] # smooth each spike as we count it
                        spikeRaster[iCell, trialTypeIx, np.int0(iTrialOfType[trialTypeIx]), spikePsthIndices[iSpike]]= spikeRaster[iCell, trialTypeIx, np.int0(iTrialOfType[trialTypeIx]), spikePsthIndices[iSpike]] + 1;  # also put the spike into the raster    
                    iTrialOfType[trialTypeIx] = iTrialOfType[trialTypeIx] + 1 
        PSTH = (1000 / self.psthResolution) * PSTC # convert spikes/binWidth to spikes/s
        
        return PSTH, PSTC, spikeRaster

