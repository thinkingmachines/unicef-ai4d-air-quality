.PHONY: clean clean-test clean-pyc clean-build dev venv help requirements-dev.txt
.DEFAULT_GOAL := help
-include .env

help:
	@awk -F ':.*?## ' '/^[a-zA-Z]/ && NF==2 {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Setup dev environment
	pip install pip-tools
	pip-sync requirements.txt

requirements:
	pip-compile -v -o requirements.txt requirements.in
	pip-sync requirements.txt

train:
	export PYTHONPATH=. && python scripts/train.py --config-path=${config-path}

air4thai-data:
	export PYTHONPATH=. && python scripts/generate_data.py \
	--locations-csv=data/2022-04-29-air4thai-th-stations.csv \
	--ground-truth-csv=data/2022-04-29-air4thai-daily-pm25.csv \
	--admin-bounds-shp=data/tha_admin_bounds_adm3/tha_admbnda_adm3_rtsd_20220121.shp \
	--hrsl-tif=data/tha_general_2020.tif \
	--start-date=2021-01-01 \
	--end-date=2021-12-31
