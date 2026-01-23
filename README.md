# PertAL

We introduce PertAL, a novel active learning framework to guide low-budget single-cell perturbation screening. To intelligently prioritize perturbations, PertAL integrates three key scoring modules: LLM-driven scoring of biological reasoning, multi-view diversity assessment, and gradient-based sensitivity quantification. By combining these scores, PertAL facilitates efficient exploration under strict budget constraints.

## 1.Overview

![Intro](Overview_of_PertAL.png)

## 2. Installation

### Create a new environment

First, create a new environment for the PertAL project. The recommended Python version is 3.8.

```bash
# Create a new environment with Python 3.8
conda create -n pertAL python=3.8
# Activate the environment
conda activate pertAL
```

### Install dependencies

Next, install the dependencies from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

This will install the necessary libraries and dependencies required to run the project.

## 3. Run

To run the program, use the provided `run.py` script. This script allows you to specify several key parameters that control the execution.

#### Key parameters

Here are the key parameters you can pass to the script:

- `--strategy` (default: `PertAL`): Strategy to use for the analysis.
- `--device` (default: `1`): Which GPU to use. Set `0` for GPU 0, `1` for GPU 1, and so on.
- `--prior_scfm_kernel` (default: `scgpt_blood`): Specify which scFM feature kernel to use.
- `--seed` (default: `5`): Set the random seed for reproducibility.
- `--dataset_name` (default: `replogle_k562`): Name of the experimental dataset (e.g., `replogle_k562`, `replogle_rpe1`).
- `--llm_name` (default: `gpt41-mini`): Specify which large language model (LLM) to use for prior knowledge generation.
- `--llm_weight` (default: `0.2`): Weight of the LLM prior.

#### Example command

Here is an example of how to run the script with the key parameters:

```shell
python run.py --strategy PertAL --device 0 --dataset_name replogle_k562 --seed 1 --prior_scfm_kernel scgpt_blood --llm_name gpt41-mini --llm_weight 0.2
```

### Datasets

Due to data size and availability restrictions, the target dataset must be manually downloaded. You can access it from the following URL: [Dataset](https://drive.google.com/drive/folders/1Hh00_cO6oRBOU6kAzhyblaJ0xhJZfyGB?dmr=1&ec=wgc-drive-globalnav-goto).
Make sure to download the dataset and place it in the appropriate directory `data` before running the program.

### Additional scripts

- **Feature Kernel Generation**: If you need to build your own feature kernel, use the `Prior_kernel_preprocess.ipynb` notebook. This notebook guides you through the process of preprocessing and generating feature kernels for the perturb-seq data.
- **LLM Prior Generation**: To generate the LLM prior, we provide the `LLM_prior_generator.py` script.
  ⚠ Make sure to replace the API keys with your personal ones in the script to avoid any issues when connecting to the LLM API.

## 4. Acknowledgements

We would like to thank the authors of the following projects for their outstanding work, which served as both inspiration and a foundation for our approach:

- The code in this project was inspired by the excellent work in [IterPert](https://github.com/Genentech/iterative-perturb-seq/tree/master) and [bmdal_reg](https://github.com/dholzmueller/bmdal_reg).

## 5. Contact us

If you have any questions or would like to learn more, feel free to reach out!
