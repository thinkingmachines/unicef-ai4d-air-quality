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
This workflow assumes you just want to train your own models on already existing datasets.

1. Get a copy of the latest dataset in CSV format from our [Google Drive folder](https://drive.google.com/drive/folders/1c4f1TJzW7uI9IgqXZY_08pJb1YvpevG1) and place it in your local `data` folder.
2. Create a yaml config file with the training configuration that you want inside the `config` folder (see `config/default.yaml` for a sample).
    * Note: this is where you specify the path to the CSV dataset from step 1.
3. Make sure your terminal's current working directory is the project root. Run the training script by running `make config-path=config/default.yaml train`, where you should replace `default.yaml` with the actual yaml file you created from step 2.
    * If you can't run Make commands on your system, you can also run the training script manually like this:
        * `export PYTHONPATH=.` (you only need to run this once per terminal sesion)
        * `python scripts/train.py --config-path=config/default.yaml`
            * Note: You can also just call the script without a config path `python scripts/train.py`, in which case it will use `config/default.yaml`.
4. The training script should have generated results in a dated folder under `data/outputs`. The folder should contain the best model and its params, metrics, SHAP charts, and the yaml config file used.


<br/>
<br/>

# 📚 Generating a training dataset
This section describes how to generate a dataset used for ML  training and evaluation. The example is based on the generation of an OpenAQ-based dataset for our experiments.

If you wish to generate your own custom dataset (e.g. generate dataset for a different year), feel free to do so - just modify the parameters accordingly.


## OpenAQ Training Dataset Example

*Notes: Make sure your terminal's current working directory is at the project root.*

1. Collect raw OpenAQ data
    * Run the collection script (this example is for collecting 2021 Thailand data through the OpenAQ API):
    ```
    export PYTHONPATH=. && \
    python scripts/collect_openaq.py \
	--start-date=2021-01-01 \
	--end-date=2021-12-31 \
	--country-code=TH
    ```
    * This script will do a bit of pre-processing and generate 2 csv files in your `data/` folder (`<timestamp>` is the datetime you ran the script):
        * `daily-pm25-<timestamp>.csv`
        * `station-list-<timestamp>.csv`
    * Feel free to rename these files.
    * *Note: Over the course of development, the OpenAQ API seems to have been under active development and we encountered intermittent errors a few times. If this happens, code has to be updated to match the API changes.*
2. Add features to the OpenAQ data
    * Take note of the filenames generated by the previous step, as they are the input to the next script for collecting features.
    * Download pre-requisite files to your local:
        * Get a copy of the contents of this [GDdrive data folder](https://drive.google.com/drive/u/0/folders/1Ni-OWGovH-4gV2VhJeao0jABMW_Wp2k_) and place them in your local `data` folder.
        * The most important here are the admin boundaries (`tha_admin_bounds_adm3/`) and population data (`tha_general_2020.tif`).
    * Sign-up for a [Google Earth Engine account](https://signup.earthengine.google.com/) if you don't have one yet, as the script uses the GEE API to collect some of the features. It will ask you to log-in when you run it.
    * Finally, run the script with the appropriate parameters. The ff. is an example, but you should change the `--locations-csv` and `--ground-truth-csv` arguments accordingly to the files generated from step 1:
    ```
    export PYTHONPATH=. && python scripts/generate_features.py \
	--locations-csv=data/2022-05-06-openaq-th-stations.csv \
	--ground-truth-csv=data/2022-05-06-openaq-daily-pm25.csv \
	--admin-bounds-shp=data/tha_admin_bounds_adm3/tha_admbnda_adm3_rtsd_20220121.shp \
	--hrsl-tif=data/tha_general_2020.tif \
	--start-date=2021-01-01 \
	--end-date=2021-12-31
    ```
    * This should generate an ML-ready file of the format: `generated_data_<timestamp>.csv` in your `data/` folder. As usual, feel free to rename the file if you wish.

# 🌍 Predicting PM2.5 levels at a target location
We provide a sample notebook for illustrating how one might use a trained model on a location in Thailand. The notebook can be found in the `notebooks/2022-05-18-prediction-example` folder. This notebook contains more explanations, and has some light EDA and viz on sample predictions for a district in Chiang Mai.

There is also a script version for just running predictions on an input CSV file of locations (the expected format of this is described in the notebook). Please run `export PYTHONPATH=. && python scripts/predict.py --help` to see details on the usage.
