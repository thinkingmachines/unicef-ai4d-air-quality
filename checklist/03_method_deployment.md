[Previous Page](02_building_the_base_table.md) | [Table of Contents](../README.md) | [Next Page](04_initial_experimentation.md)

# 03: Model/Method + Deployment Proposal
Propose models/experiments to try.

# Checkpoints
- [ ] High level architecture: Consult with MLE/ESE on scalability of proposed solution (especially if this will be scaled to a large dataset, e.g., whole country)
    - [ ] Questions to ask:
        - [ ] Scalability: Can I use the model to predict reliably on millions of data points?
        - [ ] Complexity: How long will it take to run? Days? Weeks?
        - [ ] Cost: How much will it cost to run for all data points?

# Documentation Outputs
- [ ] [Related Work](../README.md#rrl)
- [ ] Written list of methods ([Experiment Tracker](../README.md#experiment-tracker))
    - [ ] Pros and cons
    - [ ] Implementation plan
- [ ] Written resource estimates
    - [ ] Training time (and frequency)
    - [ ] Evaluation time (and frequency)
    - [ ] Cost (pipelines + tables)
    - [ ] Memory considerations
