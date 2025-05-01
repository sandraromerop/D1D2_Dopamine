# D1D2_Dopamine

This code reproduces the analysis and simulations for the main figures of the manuscript _**'Tonic dopamine and biases in value learning linked through a biologically inspired reinforcement learning model'**_ by  Romero Pinto &  Uchida, 2023 (1)

Before running the code, you should download the data. Please use the notebook **`download_data`**:
   - Downloads the data from the  [**Open Science Framework (OSF)**](https://osf.io/) repository. The repository can be viewed [here](https://osf.io/cr5mv/?view_only=bd13a2d2de1947699b56ce70610b0e9b).
   - Also includes details on the data format.
   - *No plots*

## Notebooks to reproduce figure by figure 
   - **`Figure_1`** : Reinforcement learning models
   - **`Figure_2`** : Biologically inspired reinforcement learning model
   - **`Figure_3`** : Potential mechanisms for asymmetric learning.
   - **`Figure_4`** : Habenula lesions leads to optimistic reward-seeking behavior and cue-evoked responses in dopamine neurons.
   - **`Figure_5`** : Mechanism 2 cannot explain optimistic biases in behavior and cue-evoked dopamine responses of habenula lesioned animals
   - **`Figure_6`** : Biophysical model based on firing rates of dopamine neurons predicts increases in dopamine concentration and receptor occupancies at baseline
   - **`Figure_7`** : Mechanism 1 can account for optimistic biases in reward-seeking behavior and cue-evoked dopamine responses.
   - **`Figure_8`** : Mechanism 1 predicts asymmetric learning rates in healthy humans given inter-individual differences in baseline dopamine.
   - **`Figure_Supp_1`**: Variables of model with mechanism 1 show convergence irrespective of the value of the decay factor.  
   - **`Figure_Supp_2`**: Rl model fits to the trial-by-trial anticipatory licking responses.
   - **`Figure_Supp_3_and_4`**: 
      - Signatures of distributional reinforcement learning model are preserved after habenula lesions 
      - Distributional reinforcement learning variables from the habenula lesion dataset
   - **`Figure_Supp_5`**: Cue-evoked responses in the habenula lesion data are only predicted by mechanisms 1 but not by mechanism 2
   - **`Figure_Supp_6`**: Mechanism 1 and 2 play complementary roles in distributional td learning 
   - **`Figure_Supp_7_and_8`**: 
      - Mechanism 1 predicts asymmetric learning rates and the effect of bromocriptine in healthy humans given inter-individual differences in baseline dopamine 
      - Robustness of the effect of bromocriptine in the relative reversal learning (rrl) to the choice of the drug efficiency parameter
   - **`Figure_Supp_9`**: The qualitative aspects of mechanism 1 are preserved irrespective of the assumption made about the changes in baseline dopamine caused by dopamine transients.  
   - **`Figure_Supp_10`**: Change in dopamine firing rates, dopamine concentration and receptor occupancy as a function of rpes in the linear scale or logarithmic scale.

## Notebooks to perform more in depth analysis 

2. **`tonic_da_model`**
   - Performs the basic predictions of the tonic Dopamine model proposed in the manuscript

3. **`data_analysis`**
   - Reproduces the analysis performed on the dopamine neurons from the Habenula lesion dataset (Tian & Uchida, 2015)
   - Figures from manuscript:  4b-e, 5c-d, 6b

4. **`fit_rl_behavior`**
   - Performs fits to the behavioral readout (anticipatory licking) from the Habenula lesion dataset  of reinforcement learning models
   - Figures from manuscript: 4c
  
5. **`biophysical_simulations_from_data`**
   - Performs the biophysical simulations of dopamine release and receptor occupancy, having as inputs the dopamine firing rates recorded in Tian & Uchida, 2015 (2)
   - These simulations are based on the  [codebase](https://github.com/jakobdreyer/Dopamine-Simulation-Tools) released by Jakob Dreyer.
   - *No plots*

6. **`rl_from_biophysical_simulations`**
   - Reproduces the analysis performed on the output variables from the biophysical simulations
   - Figures from manuscript: 6c, 6e-f, 6h-i

7. **`rl_simulations_from_data`**
   - Performs TD learning simulations based in the derived parameters of asymmetric learning rates from the Habenula lesion dataset and the output variables from the biophysical simulations
   - *No plots*
  
8. **`plot_rl_simulations_from_data`**
   - Performs the plots for output variables of the TD learnig simulations 
   - Figures from manuscript: 6 k-l

9. **`drugs_from_biophysical_simulations`**
   - Performs the biophysical simulations of dopamine release and receptor occupancy performed by adding an additional D2 receptor agonist (bromocriptine)
   - The simulations were done to reproduce the effects of bromocriptine on reversal learning in the study from Cools et al, 2009 (3)

10. **`plot_drugs_from_biophysical_simulations`**
    - Plots the results from the biophysical simulations of dopamine release and receptor occupancy performed by adding an additional D2 receptor agonist (bromocriptine)
    - Figures from manuscript: 7 & Extended Data Figure 6-8

  

## Auxiliary functions: 

**`rl`:**
- `agent.py`: TD learning object for performing RL simulations (deployed in `rl_simulations_from_data` )
- `tasks.py`: Task object for the Pavlovian task used in Tian & Uchida 2015 (deployed in `rl_simulations_from_data` )
- `plots.py`: Functions for plotting the outputs from the RL simulations (deployed in `plot_rl_simulations_from_data` )
- `models.py`: Contains a class and associated functions for generating the predictions of Model 1 (deployed in `tonic_da_model`)

**`biophysical_model`:**
- `dopamine_toolbox.py`: Contains the classes and functions needed to run the biophysical simulations of dopamine release and receptor occupancy. Functions are based on  [codebase](https://github.com/jakobdreyer/Dopamine-Simulation-Tools) released by Jakob Dreyer
- Additional functions were included for implementing the drug experiment simulations

**`utils`:**
- `data.py`: Functions and objects for loading and reformatting data  (deployed in `data_analysis_LHb_DA` )
- `model_fitting`:  Functions for fitting RL models to the behavioral data from Tian & Uchida 2015 ( deployed in `fit_rl_models_LHb_DA`). The functions are an extension of the ones used in   Babayan, Gershman & Uchida, 2017
- `drl.py`: Function to derive distributional RL parameters from data (deployed in `data_analysis_LHb_DA` and `drl_from_biophysical_simulations`)
- `stats_perform.py`: Functions to perform statistical tests for normality and difference in distributions (deployed in all noteboks)
- `plots.py`: miscellaneous functions for plotting
- `path.py`: path configuration for loading and saving data and analysis



## Dependencies & environment

To install the dependencies to run this code one can  take two routes:

1. With pip: Navigate to the folder `/D1D2_dopamine` in the command line and execute:
  ```
  pip install -r requirements.txt
  ```
1. Create conda environment (**recommended**):  Navigate to the folder `/D1D2_dopamine` in the command line and execute:
  ```
  conda env create -f environment.yml
  ```
  - This will create a conda environment called **'Tonic_DA'**. To execute the code from the terminal, activate the conda environment
  ```
  conda activate Tonic_DA
  ```


## References

**(1)** Romero Pinto, S., & Uchida, N. (2023). Tonic dopamine and biases in value learning linked through a biologically inspired reinforcement learning model (In preparation)


**(2)** Tian, J., & Uchida, N. (2015). Habenula Lesions Reveal that Multiple Mechanisms Underlie Dopamine Prediction Errors. Neuron, 87(6), 1304–1316


**(3)** Cools, R., Frank M.J., Gibbs S., Miyakawa A., Jagust W. & D'Esposito M.  Striatal dopamine predicts outcome-specific reversal learning and its sensitivity to dopaminergic drug administration. J. Neurosci. 29, 1538–1543 (2009)


