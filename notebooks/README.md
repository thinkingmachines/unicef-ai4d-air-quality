# Experiment notebooks

This directory contains all experimental notebooks before they're ported into
the production business logic in `src`. You're free to decide how you'll
organize the contents, but a recommended structure looks like this:

```
.
├── 2021-06-10-initial-stab-classification/
│   ├── 2021-06-10-logistic-regression.ipynb
│   └── 2021-06-11-knn.ipynb
├── 2021-06-13-testing-model-api/
│   └── 2021-06-13-test-api-endpoints.ipynb
```

where each notebook is named `YYYY-MM-DD-<unique-label>.ipynb`. They are
usually separated into folders to further segregate their concerns.

If you wish to use any of the classes or functions from your business logic in
a notebook, add this line in the first cell:

```python
import sys
sys.path.append("../../")  # include parent directory
```
