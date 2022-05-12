<div align="center">

# UNICEF AI4D Air Quality Research

</div>

<br/>
<br/>


# 📜 Description

Repository for UNICEF AI4D air quality research. Goal is to train models that can predict ground-level PM2.5 for areas with no ground-monitoring stations using satellite-derived data (e.g. Aerosol Optical Depth, Meteorological Variables, NDVI, etc) and other datasets (e.g. population).


<br/>
<br/>


# ⚙️ Local Setup for Development

Though you are free to use any python environment manager you wish, this guide will assume the usage of [miniconda](https://docs.conda.io/en/latest/miniconda.html#:~:text=Miniconda%20is%20a%20free%20minimal,zlib%20and%20a%20few%20others.).


## Requirements

1. Python 3.7+
2. make


## 🐍 One-time Set-up
Run this the very first time you are setting-up the project on a machine to set-up a local Python environment for this project.

1. Install miniconda for your environment if you don't have it yet. Either:
* Manually download and install the appropriate version from [here](https://docs.conda.io/en/latest/miniconda.html); or
* For VMs with no GUI, this is an example of how to install from your terminal:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```


2. Create a local python environment and activate it.
* Note:
    * You can change the name if you want; in this example, the env name is `ai4d-air-quality`.
```bash
conda create -n ai4d-air-quality python=3.7
conda activate ai4d-air-quality
```

3. Clone this repo and navigate into the folder. For example:
```bash
git clone git@github.com:thinkingmachines/unicef-ai4d-air-quality.git
cd unicef-ai4d-air-quality
```

4. Install the project dependencies by running:
    * Note:
        * This make command installs `pip-tools` (the python dependency manager),  `pre-commit` hooks (which enforce the automated formatters), and `jupyter`/`jupyter lab`.
        * If you don't have `make` available in your system, you can refer to the commands under `Makefile` > `dev` recipe. That is, copy-paste those commands into your terminal.
```bash
make dev
```


## 📦 Dependencies

Over the course of development, you might introduce new library dependencies. When you do so, please add it in the `requirements.*` files and include those with your commits so that other devs can get the updated list of project requirements.

For example, to add `pandas` as a dependency:

1. Add it to `requirements.in`:
```bash
# Sample requirements.in contents
numpy
pandas
```

2. Run `pip-compile` to re-generate the `requirements.txt` file.
```
pip-compile -v -o requirements.txt requirements.in
```

3. Finally, run `pip-sync` to make your local env follow `requirements.txt` exactly.
```
pip-sync requirements.txt
```

Other notes:
* Alternatively, we provide a shortcut for Steps 2 and 3 by running `make requirements`.

* Running `pip-sync requirements.txt` alone is also handy for updating your local conda env after you pull changes from GitHub, if another developer has added new requirements.


<br/>
<br/>

# 🧠 Training a Regression Model
1. Get a copy of the latest dataset in CSV format from our [Google Drive folder](https://drive.google.com/drive/folders/1c4f1TJzW7uI9IgqXZY_08pJb1YvpevG1) and place it in your local `data` folder.
2. Create a yaml config file with the training configuration that you want inside the `config` folder (see `config/default.yaml` for a sample).
    * Note: this is where you specify the path to the CSV dataset.
3. Make sure your terminal's current working directory is the project root. Run the training script by running `make config-path=config/default.yaml train`, where you should replace `default.yaml` with the actual yaml file you created from step 2.
    * If you can't run Make commands on your system, you can also run the training script manually like this:
        * `export PYTHONPATH=.` (you only need to run this once per terminal sesion)
        * `python scripts/train.py --config-path=config/default.yaml`
            * Note: You can also just call the script without a config path `python scripts/train.py`, in which case it will use `config/default.yaml`.
4. The training script should have generated results in a dated folder under `data/outputs`. The folder should contain the best model and its params, metrics, SHAP charts, and the yaml config file used.

<br/>
<br/>

# 📚 Generating a dataset with features
We provide a script for re-producing our training dataset in `scripts/generate_data.py` as a reference in case you want to understand the data collection procudure, or if you want to generate training data using your own ground truth.

This script can also be used to collect features for an arbitrary list of locations for other purposes. For example, if you'd like to predict PM2.5 for an area of interest over a certain time period.


## Reproducing the training dataset
To reproduce the training dataset in our experiments:
1. Get the (a) daily pm2.5 ground truth and (b) station info CSV files from this [Google Drive Folder](https://drive.google.com/drive/folders/1N1B8SXSkmpnwp_YTDiCJqvLSx3A3-mwC) and place them in your local `data/` folder. Raw source of this data was taken from the [Air4Thai website](http://air4thai.pcd.go.th/webV2/history/), but some manual re-formatting had to be performed since the original format wasn't convenient for automated processing, and there were minor typos.

2. Get the `tha_general_2020.tif` file from this [link](https://data.humdata.org/dataset/1ec16b2b-2a1d-4cf7-b766-0460b27b89ea/resource/c45c9659-5708-4e88-a589-6a5a8b0a0d81/download/tha_general_2020_geotiff.zip). Unzip the tif file into your `data/` folder. This HRSL population data is taken from [Humanitarian Data Exchange](https://data.humdata.org/dataset/cod-ab-tha).

3. Sign-up for a [Google Earth Engine account](https://signup.earthengine.google.com/) if you don't have one yet, as the script uses the GEE API to collect some of the features. It will ask you to log-in when you run it.

4. Run the data generation script. For example, generating the training dataset based on Air4Thai ground truth:
 `export PYTHONPATH=. && python scripts/generate_data.py --locations-csv=data/2022-04-29-air4thai-th-stations.csv --ground-truth-csv=data/2022-04-29-air4thai-daily-pm25.csv`
     * *Note: Update the CSV paths accordingly if you're using a different set of files*


5. Depending on your internet connection, the script might take around 30 minutes - 1 hour to complete for the 78 stations from Air4Thai. The generated CSV file should appear in your local `data/` folder.
