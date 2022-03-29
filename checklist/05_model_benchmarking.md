[Previous Page](04_initial_experimentation.md) | [Table of Contents](../README.md) | [Next Page](06_finalize_model.md)

# 05: Model Benchmarking
Test out all proposed methods/models on the whole dataset.

# Technical Outputs
- [ ] Python scripts in github repo
    - [ ] Model training
    - [ ] Model evaluation
    - [ ] Metric computation
- [ ] Things to check:
    - [ ] Data leakage
    - [ ] Accuracy of metric calculation
    - [ ] Model performance
    - [ ] Model explainability (which features contribute the most and why?)
    - [ ] Linting
    - [ ] Code Efficiency

# Checkpoints
Code review continuous within the team (project tech lead/DA) to check for code quality (one feature = one PR)
- [ ] Continuous code reviews
- [ ] [Optional] A task is only done if there’s already a unit test for it
- [ ] [Optional] [Check for code coverage](https://coverage.readthedocs.io/en/coverage-5.5/) (>80%)

# Documentation Outputs
- [ ] Well-organized github repo containing all the scripts for benchmarking
- [ ] README with all the instructions from data prep to model evaluation
- [ ] Documentation of performance comparison and eventual decision  ([Experiment Tracker](../README.md#experiment-tracker))
    - [ ] Feature importances
    - [ ] Rationale
    - [ ] Runtime (train/eval)
    - [ ] Memory usage
    - [ ] Size of data
