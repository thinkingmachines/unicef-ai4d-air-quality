[Table of Contents](../README.md) | [Next Page](02_building_the_base_table.md)

# 01: Data Gathering
Doing EDA on the data, finding out the level of cleanliness/dirtiness of the raw data.

## Technical Outputs
Clean Jupyter notebook/python code that produce the following:

- [ ] Data EDA
    - [ ] Data distribution
    - [ ] Number of nulls
- [ ] Data Preprocessing
    - [ ] Imputation
    - [ ] Cleaning
    - [ ] Final stats (distribution, etc.)

## Checkpoints
- [ ] Quick sense check from other team members [EDA + Preprocessing (method checking)]
- [ ] Client Project: Sign off from data owner on transformations [POCs: Might not always be able to get a sign-off, since contact persons aren’t always data owners]

## Documentation Outputs
- [ ] Well-annotated and clean notebooks/scripts
- [ ] Documentation on the data sources
    - [ ] [BQ checklist](../README.md#bq-checklist)
    - [ ] [ML Scoping](../README.md#ml-scoping)
    - [ ] [Data dictionary](../README.md#data-dictionary)

BQ Checklist [case to case basis]
- [ ] Location (i.e. BQ: project.dataset.table)
- [ ] Key fields
- [ ] Table level: what is a single row?
- [ ] Latency: how frequent is the table updated?
- [ ] Storage: how are updates done? overwrite/append?
