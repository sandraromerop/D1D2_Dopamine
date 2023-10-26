# D1D2_Dopamine

This code reproduces the analysis and simulations for the main figures of the manuscript **'Tonic dopamine and biases in value learning linked through a biologically inspired reinforcement learning model'** by  Romero Pinto &  Uchida, 2023

## Notebooks to reproduce the figures:

- **`download_data`**
  - Downloads the data from the  [**Open Science Framework (OSF)**](https://osf.io/) repository. The repository can be viewed [here](https://osf.io/cr5mv/?view_only=bd13a2d2de1947699b56ce70610b0e9b).
  - Also includes details on the data format.

- **`tonic_da_model`**
  - Performs the basic predictions of the tonic Dopamine model proposed in the manuscript

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

# TODOS:
- Sample code to produce biophysical simulations 
- Code to produce drug manipulations


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
- `path.py`: path configuration for loading and saving data and analysis

## References

**(1)** Romero Pinto, S., & Uchida, N. (2023). Tonic dopamine and biases in value learning linked through a biologically inspired reinforcement learning model (In preparation)


**(2)** Tian, J., & Uchida, N. (2015). Habenula Lesions Reveal that Multiple Mechanisms Underlie Dopamine Prediction Errors. Neuron, 87(6), 1304–1316. https://doi.org/10.1016/j.neuron.2015.08.028


