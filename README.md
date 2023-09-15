# GPT-LS: Generative Pre-Trained Transformer with Off-line Reinforcement Learning for Logic Synthesis
[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen)](https://github.com/NYU-MLDA/OpenABC) 
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Update

1) Full dataset (Initial, intermediate and final optimized AIGs) is hosted at [OpenABC-D](https://app.globus.org/file-manager?origin_id=ae7b03ad-9e50-472c-9601-ff99054ae47c&origin_path=%2F). All relevant bench, graphml, label CSVs and run logs are stored here. The entire zip is 1.4TB so it has been equally divided into 14 chunks, each of 107GB. For downloading and unzipping, minimum of 3TB disk space is required.

2) ML-ready dataset (Pytorch files for initial AIG and synthesis recipe information + target labels as pickled file ~ 19GB) is hosted [here](https://zenodo.org/record/6399454#.YkTglzwpA5k). All one need is to configure the paths as documented and run the models for QoR prediction.

3) Pytorch-geometric has newly released versions 2.0.* and is not backward compatible. Thus, to use the already existing pytorch data, please install [Pytorch-geometric](https://github.com/pyg-team/pytorch_geometric/tags) version < 2.0.* or regenerate the pt data files using the dumped graphml files.

4) Original AIGs of designs have been added in the repository under **bench_openabcd**.
 

## Overview

**GPT-LS** is a new algorithm developed to optimize logic synthesis (LS) in electronic design automation (EDA). LS is a process that transforms a high-level circuit description into a gate-level netlist, typically utilizing a unified heuristic algorithm to optimize various combinational circuits. The GPT-LS model uses decision transformer (DT), a form of offline reinforcement learning, to generate a primitive sequence (PS) that achieves design goals in a shorter time compared to traditional machine learning-based approaches. GPT-LS has been trained on a large-scale logic synthesis dataset and has achieved results that match those of previous state-of-the-art (SOTA) methods in a significantly shorter time.


## Installing dependencies

We recommend using [venv](https://docs.python.org/3/library/venv.html) or [Anaconda](https://www.anaconda.com/) environment to install pre-requisites packages for running our framework and models.
We list down the packages which we used on our side for experimentations. We recommend installing the packages using *requirements.txt* file provided in our repository.

- cudatoolkit = 10.1
- numpy >= 1.20.1
- pandas >= 1.2.2
- pickleshare >= 0.7.5
- python >=3.9
- pytorch = 1.8.1
- scikit-learn = 0.24.1
- torch-geometric=1.7.0
- tqdm >= 4.56
- seaborn >= 0.11.1
- networkx >= 2.5
- joblib >= 1.1.0

Here are few resources to install the packages (if not using *requirements.txt*)

- [Pytorch](https://pytorch.org/get-started/locally/)
- [Torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Networkx](https://networkx.org/documentation/stable/install.html)

Make sure that that the cudatoolkit version in the gpu matches with the pytorch-geometric's (and dependencies) CUDA version.

## Organisation

### Dataset directory structure

	├── OPENABC_DATASET
	│   ├── bench			# Original and synthesized bench files. Log reports post technology mapping
	│   ├── graphml                # Graphml files
	│   ├── lib			# Nangate 15nm technology library
	│   ├── ptdata			# pytorch-geometric compatible data
	│   ├── statistics		# Area, delay, number of nodes, depth of final AIGs for all designs
	│   └── synScripts		# 1500 synthesis scripts customized for each design

1. In ```bench``` directory, each design has a subfolder containing original bench file: design_orig.bench, a log folder containing log runs of 1500 synthesis recipes, and syn<N>.zip file containing all bench files synthesized with synthesis recipe N.

2. In ```graphml``` directory, each design has subfolder containing zipped graphml files corresponding to the bench files created for each synthesis runs.

3. In ```lib``` directory, Nangate15nm.lib file is present. This is used for technology mapping post logic minimization.

4. In ```ptdata``` directory, we have subfolders for each design having zipped pytorch file of the format *designIP_synthesisID_stepID.pt*. Also, we kept train-test split csv files for each learning tasks in subdirectories with naming convention *lr_ID*.

5. In ```statistics``` diretcory, we have two subfolders: ```adp``` and ```finalAig```. In ```adp```, we have csv files for all designs with information about area and delay of final AIG post tech-mapping. In ```finalAig```, csv files have information about graph characteristics of final AIGs obtained post optimization. Also, there is another file named *synthesisstastistics.pickle* which have all the above information in dictionary format. This file is used for labelling purpose in ML pipeline for various tasks.

6. In ```synScripts``` directory, we have subfolders of each design having 1500 synthesis scripts.


### Data generation

	├── datagen
	│   ├── automation 			      # Scripts for automation (Bulk/parallel runs for synthesis, AIG2Graph conversions etc.)
	│   │   ├── automate_bulkSynthesis.py         # Shell script for each design to perform 1500 synthesis runs
	│   │   ├── automate_finalDataCollection.py   # Script file to collect graph statistics, area and delay of final AIG
	│   │   ├── automate_synbench2Graphml.py      # Shell script file generation to involking andAIG2Graphml.py
	│   │   └── automate_synthesisScriptGen.py    # Script to generate 1500 synthesis script customized for each design
	│   └── utilities
	│       ├── andAIG2Graphml.py		      # Python utility to convert AIG BENCH file to graphml format
	│       ├── collectAreaAndDelay.py            # Python utility to parse log and collect area and delay numbers
	│       ├── collectGraphStatistics.py         # Python utility to for computing final AIG statistics
	│       ├── pickleStatsForML.py               # Pickled file containing labels of all designs (to be used to assign labels in ML pipeline)
	│       ├── PyGDataAIG.py		      # Python utility to convert synthesized graphml files to pytorch data format
	│       └── synthID2SeqMapping.py	      # Python utility to annotate synthesis recipe using numerical encoding and dump in pickle form

1. ```automation``` directory contains python scripts for automating bulk data generation (e.g. synthesis runs, graphml conversion, pytorch data generation etc.). ```utilities``` folder have utility scripts performing various tasks and called from automation scripts.

2. ```Step 1```: Run *automate_synthesisScriptGen.py* to generate customized synthesis script for 1500 synthesis recipes. One can see the template of a synthesis recipe in ```referenceScripts``` under ```synScripts``` folder.

3. ```Step 2```: Run *automate_bultkSynthesis.py* to generate a shell script for a design. Run the shell script to perform the synthesis runs. Make sure **yosys-abc** is available in **PATH**.

4. ```Step 3```: Run *automate_synbench2Graphml.py* to generate a shell script for generating graphml files. The shell script invokes *andAIG2Graphml.py* using 21 parallel threads processing data of each synthesis runs in sequence.

5. ```Step 4```: Run *PyGDataAIG.py* to generate pytorch data for each graphml file of the format *designIP_synthesisID_stepID.pt*.

6. ```Step 5```: Run *collectAreaAndDelay.py* and *collectGraphStatistics.py* to collect information about final AIG's statistics. Post that, run *pickleStatsForML.py* which will output *synthesisStatistics.pickle* file.

7. ```Step 6```: Run *synthID2SeqMapping.py* utility to generate *synthID2Vec.pickle* file containing numerically encoded data of synthesis recipes.


### Benchmarking models: Training and evaluation

	├── models
	│   ├── classification
	│   │   └── ClassNetV1
	│   │       ├── model.py			# Graph convolution network based architecture model
	│   │       ├── netlistDataset.py		# Dataset loader
	│   │       ├── train.py			# Train and evaluation utility
	│   │       └── utils.py			# Utitility functions
	│   └── qor
	│       ├── NetV1
	│       │   ├── evaluate.py
	│       │   ├── model.py
	│       │   ├── netlistDataset.py
	│       │   ├── train.py
	│       │   └── utils.py
	│       ├── NetV2
	│       │   ├── evaluate.py
	│       │   ├── model.py
	│       │   ├── netlistDataset.py
	│       │   ├── train.py
	│       │   └── utils.py
	│       └── NetV3
	│           ├── evaluate.py
	│           ├── model.py
	│           ├── netlistDataset.py
	│           ├── train.py
	│           └── utils.py

```models``` directory contains the benchmarked model described in details in our paper. The names of the python utilities are self explainatory.

#### Case 1: Prediction QoR of a synthesis recipe

We recommend creating a following folder hierarchy before training/evaluating a model using our dataset and model codes:

	├── OPENABC-D
	│   ├── lp1
	│   │   ├── test_data_set1.csv
	│   │   ├── test_data_set2.csv
	│   │   ├── test_data_set3.csv
	│   │   ├── train_data_set1.csv
	│   │   ├── train_data_set2.csv
	│   │   └── train_data_set3.csv
	│   ├── lp2
	│   │   ├── test_data_set1.csv
	│   │   └── train_data_set1.csv
	│   ├── processed
	│   ├── synthesisStatistics.pickle
	│   └── synthID2Vec.pickle


```OPENABC-D``` is the top level directory containing the dataset, train-test split files, and labeled data available. Transfer all the relevant zipped pytorch data in the subdirectory ```processed```.

The user can now go the ```models``` directory and run codes for training and evaluation. An example run for **dataset split strategy 1** (Train on first 1000 synthesis recipe, predict QoR of next 500 recipe)

```
python train.py --datadir $HOME/OPENABC-D --rundir $HOMEDIR/NETV1_set1 --dataset set1 --lp 1 --lr 0.001 --epochs 60 --batch-size 32

```

Setting ```lp=1``` and ```dataset=set1``` will pick appropriate train-test split strategy dataset for QoR regression problem. The model will run for 60 epochs and report the training, validation and test performance on the dataset outputing appropriate plots.

Similarly for **split-strategy 2** and **3**, one can set the dataset as ```set2``` and ```set3``` respectively.

For evaluating performance of specific model on a custom curated dataset, a user can create appropriate csv file with dataset instances and add it to dictionary entry in ```train.py```. For evaluating existing dataset split, one can run the following code.

```
python evaluate.py --datadir $HOME/OPENABC-D --rundir $HOMEDIR/NETV1_set1 --dataset set1 --lp 1 --model "gcn-epoch20-loss-0.813.pt" --batch-size 32

``` 

The test-MSE performance we obtained on our side are as follows:

|   Net Type |     Case-I     |   Case-II     |   Case-III   |
|    :---:   |     :---:      |    :---:      |    :---:     |
|    NetV1   |   0.648+-0.05  |  10.59+-2.78  |  0.588+-0.04 |
|    NetV2   |   0.815+-0.02  |  1.236+-0.15  |  0.538+-0.01 | 
|    NetV3   |   0.579+-0.02  |  1.470+-0.14  |  0.536+-0.03 |

<a id='Citation'></a>

## How to cite

If you use this code/dataset, please cite:

```
@misc{chowdhury2021openabcd,
      title={OpenABC-D: A Large-Scale Dataset For Machine Learning Guided Integrated Circuit Synthesis}, 
      author={Animesh Basak Chowdhury and Benjamin Tan and Ramesh Karri and Siddharth Garg},
      year={2021},
      eprint={2110.11292},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```




















