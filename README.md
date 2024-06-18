
## License
This file is part of the project MEGSubnetworks. All code in MEGSubnetworks is free: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License on this link. You should have received a copy of the GNU General Public License along with MEGSubnetworks. If not, see [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).
## Status
Manuscript has been submitted.

## Objective
1) To identify people based on sub-network patterns of a functional connectome, maximizing fingerpinting metrics using Simulated Annealing optimization 2) To differentiate neurodegenerative diseases from healthy controls using these sub-networks with deep learning algorithms.

## Collaborators
Vasiles Balabanis conducted all the analysis and was both first and corresponding author.

## Supervisors
Professor [Jiaxiang Zhang](https://www.swansea.ac.uk/staff/jiaxiang.zhang/), Professor [Xianghua Xie](https://www.swansea.ac.uk/staff/x.xie/) and Dr [Su Yang](https://www.swansea.ac.uk/staff/su.yang/) from the Department of Computer Science in Swansea University.
## Dataset
The dataset we used was for the Simulated Annealing optimization was collected in the CUBRIC facility, in Cardiff University and was multi-session resting-state data of 43 subjects. The dataset we used for disease differentiation of 30 Parkinson's disease patients and 30 healthy controls of single-session resting-state data was from the open-access [OMEGA](https://www.mcgill.ca/bic/neuroinformatics/omega) dataset.

## Contents
The codes listed below are intended to allow other researchers to find optimal sub-networks and differentiate diseases using MEG data.
## 1-Disease Differentation.ipynb 
Pipeline:
1. Load MEG functional connectomes in format `(num_subjects, num_regions, num_regions)`
2. Load sub-networks in format `(num_sub_networks, num_regions)`
3. Extract sub-networks from diseased and healthy subjects' functional connectomes
4. Prepare data, where diseased and healthy are separated, standardized and assigned binary labels
5. Train dual-objective stacked autoencoder across k-folds and sub-network feature lengths.
6. Extract results, in terms of accuracies per epoch, per cross-validation, per feature length and final predictions, per cross-validation, per feature-length.

## 2-SimulatedAnnealing.py 
Pipeline:
1. Load four MEG functional connectomes (FCs) in format `(num_subjects, num_regions, num_regions)`, where two FCs are segments from one session, and two FCs are segments from another session
2. Initialize a number of regions in the sub-network to take and number of Simulated Annealing runs to run.
3. Initialize a random region for sub-network configuration
4. Run the Simulated Annealing algorithm, which swaps random regions while trying to converge to better solutions
5. Extract results, in terms of performance historical convergence and found sub-network regions.

## 3-submitToSlurm.sh 
Pipeline:
1. Setup SimulatedAnnealing.py file and conda environment on Slurm.
2. Run the optimization using job-array
3. Extract results, in terms of performance historical convergence and found sub-network regions from the Simulated Annealing.

## Dependencies Setup Instructions

To set up a Python development environment using `conda` and install necessary libraries like Jupyter Notebook, NumPy, TensorFlow, and Scikit-learn, follow the instructions below.

### Requirements for deep learning model:

1. **Anaconda or Miniconda** installed on your system. You can download it from:
   - [Anaconda](https://www.anaconda.com/products/distribution)
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. **Other Python Environment:** If you prefer not to use Conda, you can use `virtualenv` or `venv` to create isolated Python environments:
   - [Virtualenv Guide](https://docs.python-guide.org/dev/virtualenvs/)
   - [Venv Guide](https://docs.python.org/3/library/venv.html)

### Requirements for Simulated Annealing:
1. **Anaconda or Miniconda** installed on your system. You can download it from:
   - [Anaconda](https://www.anaconda.com/products/distribution)
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. **Other Python Environment:** If you prefer not to use Conda, you can use `virtualenv` or `venv` to create isolated Python environments:
   - [Virtualenv Guide](https://docs.python-guide.org/dev/virtualenvs/)
   - [Venv Guide](https://docs.python.org/3/library/venv.html)
#### Optional: SLURM
- **SLURM Workload Manager**: SLURM ideally should be installed and configured on your system. This is typically available on High-Performance Computing (HPC) systems. For more information on SLURM and its installation, you can refer to the [official documentation](https://slurm.schedmd.com/documentation.html).

## Instructions for deep learning model:

#### Steps:

1. **Install Conda:**
   Download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already.

2. **Create a Conda Environment:**
   Open your command prompt or terminal and run:
   ```sh
   conda create --name myenv python=3.10.9 pip
   
3. **To activate your environment:**
   Run this:
   ```sh
   conda activate myenv
   
4. **Install packages for deep learning model**
   ```sh
   pip install numpy tensorflow scikit-learn seaborn matplotlib

5. **Using Jupyter Notebook to run the code or convert the .ipynb file to .py**
   Run the command below on your conda terminal to use Jupyter notebook. Navigate to the .ipynb file and run using the interface provided.
   ```sh
   jupyter notebook
   ```
   Alternatively, convert the .ipynb to a .py using:
   ```sh
   jupyter nbconvert --to script "Disease Differentiation.ipynb"
   ```
   and run as follows:
   ```sh
   python "Disease Differentiation.py"
   ```
6. **Collect files**
   Two numpy files will be produced. The first `autoencoderSubnetworkPredictions.npy`, representing final predictions of a model are in form `(number_of_folds, number_of_feature_subsets)`, where a nested list of `(num_of_diseased + num_of_healthy)*feature-size` test predictions as 0 or 1 is stored. The second `autoencoderSubnetworkEpochAccuracies.npy` representing accuracy changes during training is in form `(number_of_folds, number_of_feature_subsets, epoch)`. Only the test accuracy is stored in this numpy.
## Instructions for Simulated Annealing optimization:

1. **Install Conda:**
   Download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already.

2. **Create a Conda Environment:**
   Open your command prompt or terminal and run:
   ```sh
   conda create --name myenv
   
3. **To activate your environment:**
   Run this:
   ```sh
   conda activate myenv
   
4. **Install packages for Simulated Annealing**
   ```sh
   pip install argparse numpy
   
5. **Run the .py file**
  - To run the Python script directly (without SLURM), you can use:
     ```bash
     python SimulatedAnnealing.py number_of_runs number_of_regions your_file_name
     ```
   - To run the script using SLURM, use the `submitToSlurm.sh` file in this repository.

6. **Collect files**
   Numpy files will be produced in form `(number_of_runs, 5)`. The name is based on the name provided in step 5, with job-array numbers associated in the slurm code. There is an option to change the name in the Simulated Annealing code, which is by default starting all numpy file names with the string 'allRuns'. The first four indices represent the final solution's 1) sub-network regions, 2) fitness score, 3) differential identifiability, 4) number of classifications. The last index is a nested list of all accepted solutions throughout the run of Simulated Annealing, further storing the a) sub-network regions, b) differential identifiability, c) number of classifications of all previous iterations.

## Additional information on content:

### Simulated Annealing:
To use the Simulated Annealing, I recommend parallel computing, preferably a supercomputer. It will take a very long time to run even with parallel computing (6-12 hours per 2500 instances of Simulated Annealing on 40 cores). You will need a recent version of python to run. I used Python 3.11.0. Additionally, you will need to provide functional connectome numpy files, in form (subjects, regions, regions). You will need two different functional connectome files, representing different segments in time for the Simulated Annealing to work, as it cross-validates between multiple segments. In my paper, I also filter out solutions that do not succeed on a third unseen segment.

![image](https://github.com/VasilesBalabanis/MEGSubnetworkCode/assets/172070528/863265dd-0e20-47de-8a3f-011c49e7e9c1)
![image](https://github.com/VasilesBalabanis/MEGSubnetworkCode/assets/172070528/73176d59-ea13-475b-9bfb-6e77c34c4e3c)


### Deep learning model architecture:

To use the dual-objective stacked autoencoder, simply follow the notebook I have provided. You will need subjects and patients in the form (subjects, regions, regions). The pipeline is designed so that there is an equal number of healthy and diseased. Additionally, it is calibrated to 6-fold cross-validation, for 30 patients and 30 healthy and 10 regions of 90 in the AAL atlas. You will need to change variables of these in the deep learning code if you are using different numbers of subjects or different numbers of regions.

I did not optimize this model thoroughly. You do not need a GPU to run this. It takes me between 12-48 hours to complete without a GPU.

![image](https://github.com/VasilesBalabanis/MEGSubnetworkCode/assets/172070528/874cba75-19b0-4b23-839d-361a51d4ca86)
