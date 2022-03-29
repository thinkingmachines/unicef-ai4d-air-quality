<div align="center">

# Machine Learning Project Workflow Template

<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7.1+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

🚀 A workflow template for Machine Learning (ML) projects that provides the step-by-step ML project checklist as well as the supporting code templates, notebooks, and infrastructure files.

Click on [<kbd>Use this template</kbd>](https://github.com/thinkingmachines/ml-workflow-template/generate) to initialize new repository. Update [Project Resources](#project-resources) section of this README for relevant links in your project. Make sure to update the `Project Name`.

</div>
<br /><br />

# 🔧 How to Use this Template

## 📚 Links to Resources

The files linked under the [Resources section](#-resources) below serve as **samples** you can use to document certain parts of your ML project. They give a basic outline of the common things noted down when it comes to scoping the project, understanding data tables, tracking experiments, etc. Feel free to modify them as needed.

**In order to use the resources,** make a copy of the files and copy them to your project's Google Drive. Remember to update the links below to refer to your project's copy.

## ✅ The Checklist

The [Workflow Checklist section](#-workflow-checklist) serves as a **guide** to help you determine your next steps at any given stage of your ML project. This template provides a **base or stock** checklist that you can use as is. Feel free to add items as needed.

**In order to mark an item as done,** you need to edit the `.md` file in Github (or in your IDE of choice) then change the checkbox from:

`[ ]` to `[X]`

## 📦 Dependencies
- The repo uses [Poetry](https://github.com/python-poetry/poetry) to manage project dependencies.

### Before development (one-time setup)
1. To install Poetry system-wide on Linux, run the following on your terminal:
    ```bash
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
    ```
   For other operating systems, check the steps on the [Poetry
   repo](https://github.com/python-poetry/poetry#installation).
1. Update your project's metadata. In `pyproject.toml`, update the `name`, `description`, and `authors` fields.
1. Create a virtual environment and install dependencies:
    ```bash
    make dev
    ```
    Poetry will automatically create a virtual environment inside `.venv/` &ndash; you do *not* need to initialize your own!

1. Activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```

### During development
- To add dependencies:
    ```bash
    poetry add <dependency>
    ```

- To update dependencies:
    ```bash
    poetry update
    ```

- Find other workflows and FAQs at the [Poetry TM wiki](https://wiki.tm8.dev/doc/poetry-exIJa15ukh).

### Dependabot
The repo uses the [dependabot](https://docs.github.com/en/code-security/supply-chain-security/keeping-your-dependencies-updated-automatically/enabling-and-disabling-version-updates)
integration in Github to keep packages updated. To disable the integration, delete `.github/dependabot.yml`. Dependabot will check for version updates daily and will file PRs when updates are available.

### Other Notes
Do not delete or modify the [`.github.tracking`](.github.tracking) file. This file is used to check what repos have used the ML workflow template.

## 📊 DVC Setup

To install DVC and set up your remote storage, make sure you have setup the project and GCP name in `.env`.
Run:
```
make dvc
```
*This has to be done only once.*

More information on DVC [here](https://wiki.tm8.dev/doc/data-version-control-dvc-lMqfvLWuCa).

## 🔄 Getting the Latest Updates

If your project uses this repository as a template, you can pull the latest ML Workflow Template updates by performing the following steps.

1. **In your project's repository**, create a temporary branch `merge-template-updates` (or any other branch name you deem appropriate) from your repository's `main` or `master` branch.
    ```
    git checkout -b merge-template-updates
    ```
1. Add `ml-workflow-template` as a remote.
    ```
    git remote add ml-workflow-template git@github.com:thinkingmachines/ml-workflow-template.git
    ```
1. Fetch latest updates from ML Workflow Template using the remote that was just added.
    ```
    git fetch ml-workflow-template
    ```
1. Merge in the latest updates and fix conflicts if any.
    ```
    git merge --allow-unrelated-histories --squash ml-workflow-template/main
    ```
1. Create a pull request from `merge-template-updates` to `main` or `master`.

<br />
<br />
<br />
<br />


**DELETE EVERYTHING ABOVE FOR YOUR PROJECT**

---

<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7.1+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

</div>

# 📜 Description

Project description here.

# ⚙️ Setup

## Requirements

1. Python 3.7.1+
1. make
1. [Poetry](https://github.com/python-poetry/poetry)

## Development
1. To install Poetry system-wide on Linux, run the following on your terminal:
    ```bash
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
    ```
    For other operating systems, check the steps on the [Poetry
   repo](https://github.com/python-poetry/poetry#installation).

1. To create a virtual environment and install dependencies, `cd` into `<project-name-here>` and run:
    ```bash
    make dev
    ```
1. To activate the virtual environment, run:
    ```bash
    source .venv/bin/activate
    ```

## Environment Variables

```
cp -i .env.example .env
```

Edit the `.env` and `.env.example` files based on your project needs.

# ♻️  Workflows

## Running tests
Ensure that tests exist in the `tests` directory. To execute all checks, run:

```bash
make test
```

# 📚 Resources

(Don't forget to update the links once you've created your own copy!)
* <a href="https://drive.google.com/drive/u/1/folders/1e128qapNrWixt3ckqSJStEdEtY3iQwR-" name="gdrive">Project Google Drive</a>
* <a href="https://docs.google.com/document/d/1Zl9CB9aaa28QwhnFjhJzL2c_k-8mOle5claX9O2k0Eo/edit?usp=sharing" name="ml-scoping">ML Scoping</a>
* <a href="https://docs.google.com/spreadsheets/d/1QMLmrB2A_Rsl8rP6wQHcgkufZgL9vNFniSdZZPNj5XY/edit?usp=sharing" name="bq-checklist">BQ Checklist</a>
* <a href="https://docs.google.com/spreadsheets/d/1YYahh_RYJaWzVRk8uSuNm9aYAAeksnqc1B-yfzVaPXI/edit?usp=sharing" name="data-dictionary">Data Dictionary</a>
* <a href="https://docs.google.com/spreadsheets/d/1JFBZhAq1nvcqFtafxMFNg342GjLBJIUcHCLqCPgAl88/edit?usp=sharing" name="experiment-tracker">Experiment Tracker</a>
* <a href="https://docs.google.com/document/d/1pYHLuhF3ogpxQYBuatuHRiYmcSiv9tjfe7pYew1y_tA/edit?usp=sharing" name="rrl">Review of Related Literature</a>

# ✅ Workflow Checklist

1. [Data Gathering](checklist/01_data_gathering.md)
1. [Building the Base Table](checklist/02_building_the_base_table.md)
1. [Model/Method + Deployment Proposal](checklist/03_method_deployment.md)
1. [Initial Experimentation](checklist/04_initial_experimentation.md)
1. [Model Benchmarking](checklist/05_model_benchmarking.md)
1. [Finalize Model](checklist/06_finalize_model.md)
1. [Present Results to Client](checklist/07_present_results.md)
1. [Iterate on the Model](checklist/08_model_iteration.md)
1. [Scale/Deploy the Model](checklist/09_scale_and_deploy_model.md)
1. Persistent: [Documentation](checklist/10_documentation.md)
