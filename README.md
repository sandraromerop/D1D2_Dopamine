# D1D2_Dopamine

This code reproduces the analysis and simulations for the main figures of the manuscript **'Tonic dopamine and biases in value learning linked through a biologically inspired reinforcement learning model'** by  Romero Pinto &  Uchida, 2023

## Notebooks to reproduce the figures:

- **`data_analysis_LHb_DA`**
  - Reproduces the analysis performed on the dopamine neurons from the Habenula lesion dataset (Tian & Uchida, 2015)
  - Figures from manuscript:  4b-e, 5c-d, 6b

- **`fit_rl_models_LHb_DA`**
  - Performs fits to the behavioral readout (anticipatory licking) from the Habenula lesion dataset  of reinforcement learning models
  - Figures from manuscript: 4c
  
- **`drl_from_biophysical_simulations`**
  - Reproduces the analysis performed on the output variables from the biophysical simulations
  - Figures from manuscript: 6c, 6e-f, 6h-i

- **`rl_simulations_from_data`**
  - Performs TD learning simulations based in the derived parameters of asymmetric learning rates from the Habenula lesion dataset and the output variables from the biophysical simulations
  - *No plots*
  
- **`plot_rl_simulations_from_data`**
  - Performs the plots for output variables of the TD learnig simulations 
  - Figures from manuscript: 6 k-l

## Auxiliary functions: 

**`rl`:**
- `agent.py`: TD learning object for performing RL simulations (deployed in `rl_simulations_from_data` )
- `tasks.py`: Task object for the Pavlovian task used in Tian & Uchida 2015 (deployed in `rl_simulations_from_data` )
- `plots.py`: Functions for plotting the outputs from the RL simulations (deployed in `plot_rl_simulations_from_data` )
  
**`utils`:**
- `data.py`: Functions and objects for loading and reformatting data  (deployed in `data_analysis_LHb_DA` )
- `model_fitting`:  Functions for fitting RL models to the behavioral data from Tian & Uchida 2015 ( deployed in `fit_rl_models_LHb_DA`). The functions are an extension of the ones used in   Babayan, Gershman & Uchida, 2017
- `drl.py`: Function to derive distributional RL parameters from data (deployed in `data_analysis_LHb_DA` and `drl_from_biophysical_simulations`)
- `stats_perform.py`: Functions to perform statistical tests for normality and difference in distributions (deployed in all noteboks)
- `plots.py`: miscellaneous functions for plotting


## Data format: 

**`unit['data']`: Habenula lesion dataset**

- `unit['data']['TrialTypes']` : Index of trial type per trial  (Size: N trials)
- `unit['data']['TrialNames']` : Name of trial type per index (Size: N trialypes )

- `unit['data']['responses']`: **Responses from the habenula lesion dataset**

  - `unit['data']['responses']['lick']` : Timestamp of licks (Size: N licks per session)
  - `unit['data']['responses']['spike']`: Timestamp of spikes (Size: N spikes per session)

- `unit['data']['events']`: **Events in session from the habenula lesion dataset**

  - `unit['data']['events']['odorOn']`: Timestamp of odor onset per trial (Size: N trials)
  - `unit['data']['events']['odorOff']`: Timestamp of odor offset per trial (Size: N trials)
  - `unit['data']['events']['airpuffOn']`: Timestamp of airpuff onset per trial (Size: N trials)
  - `unit['data']['events']['rewardOn']`: Timestamp of reward onset per trial (Size: N trials)
  - `unit['data']['events']['trialStart']`: Timestamp of trialstart per trial (Size: N trials)
  - `unit['data']['events']['odorID']`: Index of odor per trial (Size:  N trials)

**`unit['simulation_results']`: Simulation results from the biophysical simulations based on data**

  - `unit['simulation_results']['da_conc']` : DA concentrarion per trial (Size: Timestamps per trial x N trials)
  - `unit['simulation_results']['d1_occ']` : D1 occupancy per trial (Size: Timestamps per trial x N trials)
  - `unit['simulation_results']['d2_occ']` : D2 occupancy per trial (Size: Timestamps per trial x N trials)
  - `unit['simulation_results']['input_fr']` : Input firing rate per trial (Size: Timestamps per trial x N trials)
  - `unit['simulation_results']['time_ax']` : time axis per trial (Size: Timestamps per trial x N trials)
