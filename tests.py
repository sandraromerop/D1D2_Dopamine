#%%
import scipy.io as scio
import os
import glob

groups = ['Control','Lesion']
iCell = 0

g_dir= '/Users/sromeropinto/Library/CloudStorage/GoogleDrive-sromeropinto@g.harvard.edu/My Drive/Data/LHb/Ju_LHb_Lesions/'
data_path = os.path.join(g_dir,  'lightidunits',groups[0])
file_list = glob.glob(data_path + '/*formatteda.mat')
unit = scio.loadmat(file_list[iCell],squeeze_me=True,struct_as_record=False)
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
