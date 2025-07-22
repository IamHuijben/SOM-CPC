### SOM-CPC
This repository contains all code to reproduce the paper "SOM-CPC: Unsupervised Contrastive Learning with Self-Organizing Maps for Structured Representations of High-Rate Time Series", which was published at ICML, 2023.

Moreover, in experiments/SOM_CPC/2025_JournNeuroscienceMethods_experiments it contains the exact settings for the models that were run in the paper "Deep clustering of polysomnography data to characterize sleep structure in healthy sleep and non-rapid eye movement parasomnias", published in Journal of Neuroscience Methods, 2025.

#### Contact:
If you need some help or you want to discuss a possible collaboration, feel free to get into contact via: iris.huijben@maastrichtuniversity.nl 

#### Dependencies:

Download the anaconda package (https://www.anaconda.com/) and the somperf library (https://github.com/FlorentF9/SOMperf).

In the anaconda prompt run:

conda create -n SOM_CPC_env python==3.7.10
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r SOM_CPC_requirements.txt

Go to the folder where you stored the somperf library folders and run: python setup.py install
Then activate the environment with: conda activate SOM_CPC_env


#### Running an experiment

##### Data processing
The physiological and audio experiments start with running preprocessing/physiological_data_main.py or preprocessing/audio_main.py scripts to acquire the data in the right format. 
Both datasets first need to be downloaded locally to run these scripts.
The toy experiments do not need preprocessing anymore, as we published the generated dataset in the data/random_walk_freq_sinusoids_processed folder. However, to rerun or change preprocessing, you can use preprocessing/synthetic_data_main.py

##### Running a model
After having the data in place, you can train one of the different models from the experiments/<model_type>/<experiment_type>/main.py files for the different models and experiments. 
The trained model will automatically be saved in your working directory.
Note that training the SOM, supervised classifier, or running the fit_pca_kmeans.py script require an already trained CPC model to run on top of.

All model folders contain experiment_specific.yml files that specify all the settings as used in the paper.

##### Running inference
To run inference on any of the models that include a SOM, you can run the scripts in the inference/ folder. To evaluate performance with K-means (and PCA), you can use the fit_pca_kmeans.py script in the experiment folder directly, as it fits PCA/K-means on the training set and directly performs inference on the test set.

#### Citation
Please cite the following paper if you find this code useful in your work:

```
@inproceedings{huijben2023,
  title={Som-cpc: Unsupervised Contrastive Learning with Self-Organizing Maps for Structured Representations of High-Rate Time Series},
  author={Huijben, Iris AM and Nijdam, Arthur Andreas and Overeem, Sebastiaan and Van Gilst, Merel M and Van Sloun, Ruud},
  booktitle={International Conference on Machine Learning},
  pages={14132--14152},
  year={2023},
  organization={PMLR}
}
```

If you are interested in using SOM-CPC for discovering structure in polysomnography data and EEG data during sleep specificially, have a look at:
```
@inproceedings{huijben2025,
  title={Deep clustering of polysomnography data to characterize sleep structure in healthy sleep and non-rapid eye movement parasomnias},
  author={Huijben, Iris AM and van Sloun, Ruud JG and Pijpers, Angelique and Overeem, Sebastiaan, and van Gilst, Merel M},
  journal = {Journal of Neuroscience Methods},
  volume = {423},
  pages = {110516},
  year = {2025},
  issn = {0165-0270},
  doi = {https://doi.org/10.1016/j.jneumeth.2025.110516},
  url = {https://www.sciencedirect.com/science/article/pii/S0165027025001608},
}
```

