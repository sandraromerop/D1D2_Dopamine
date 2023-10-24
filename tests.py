#%%
import scipy.io as scio
import os
import glob
import pickle

groups = ['Control','Lesion']
iCell = 0

# g_dir= '/Users/sromeropinto/Library/CloudStorage/GoogleDrive-sromeropinto@g.harvard.edu/My Drive/Data/LHb/Ju_LHb_Lesions/'
# data_path = os.path.join(g_dir,  'lightidunits',groups[0])
# file_list = glob.glob(data_path + '/*formatteda.mat')
# unit = scio.loadmat(file_list[iCell],squeeze_me=True,struct_as_record=False)
#%%

ig = 0
id_unit = 0
g_dir = '/Users/sromeropinto/Dropbox (Uchida Lab)/Sandra Romeo/Manuscript D1D2/data/pickle_files'
data_path = os.path.join(g_dir, groups[ig])
file_list = glob.glob(data_path + "/*.pickle")
with open(file_list[id_unit], 'rb') as handle:
    unit = pickle.load(handle)


# unit['data']['TrialTypes'] : Index of trial type per trial  (Size: N trials)
# unit['data']['TrialNames'] : Name of trial type per index (Size: N trialypes )

# unit['data']['responses']: Responses from the habenula lesion dataset
# unit['data']['responses']['lick'] : Timestamp of licks (Size: N licks per session)
# unit['data']['responses']['spike']: Timestamp of spikes (Size: N spikes per session)
# unit['data']['events']: Events in session from the habenula lesion dataset
# unit['data']['events']['odorOn']: Timestamp of odor onset per trial (Size: N trials)
# unit['data']['events']['odorOff']: Timestamp of odor offset per trial (Size: N trials)
# unit['data']['events']['airpuffOn']: Timestamp of airpuff onset per trial (Size: N trials)
# unit['data']['events']['rewardOn']: Timestamp of reward onset per trial (Size: N trials)
# unit['data']['events']['trialStart']: Timestamp of trialstart per trial (Size: N trials)
# unit['data']['events']['odorID']: Index of odor per trial (Size:  N trials)

# unit['simulation_results']: Simulation results from the biophysical simulations based on the habenula lesion dataset
# unit['simulation_results']['da_conc'] : DA concentrarion per trial (Size: Timestamps per trial x N trials)
# unit['simulation_results']['d1_occ'] : D1 occupancy per trial (Size: Timestamps per trial x N trials)
# unit['simulation_results']['d2_occ'] : D2 occupancy per trial (Size: Timestamps per trial x N trials)
# unit['simulation_results']['input_fr'] : Input firing rate per trial (Size: Timestamps per trial x N trials)
# unit['simulation_results']['time_ax'] : time axis per trial (Size: Timestamps per trial x N trials)

#%%

# unit['S'].TrialTypes
# unit['S'].TrialNames
#%%
# unit['S'].responses:
# unit['S'].responses.lick
# unit['S'].responses.spike
#%%
# unit['S'].events
# unit['S'].events.odorOn
# unit['S'].events.odorOff
# unit['S'].events.airpuffOn
# unit['S'].events.rewardOn
# unit['S'].events.trialStart
# unit['S'].events.odorID
#%%
# dir(unit['simulation_results'])
# unit['simulation_results'].da_conc
# unit['simulation_results'].d1_occ
# unit['simulation_results'].d2_occ
# unit['simulation_results'].input_fr
# unit['simulation_results'].time_ax


#%%


#  'analytical_meanDA',
#  'analytical_stdDA',
#  'd1_ac5',
#  'd1_camp',
#  'd1_meanAC5',
#  'd1_meancAMP',
#  '',
#  'd1_tonic_meanAC5',
#  'd1_tonic_meancAMP',
#  'd2_ac5',
#  'd2_camp',
#  'd2_meanAC5',
#  'd2_meancAMP',
#  '',
#  'd2_tonic_meanAC5',
#  'd2_tonic_meancAMP',
#  '',
#  'dac5',
#  'i_phasic',
#  '',
#  'meanDA',
#  'mean_input_fr',
#  'synch',
#  '']
