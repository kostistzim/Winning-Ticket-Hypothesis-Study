# Winning-Ticket-Hypothesis-Study

A collection of experiments, code and results used to study the Lottery Ticket / Winning Ticket Hypothesis across several model families (CNNs, LeNet, VGG, and BERT). The repository contains training, pruning and evaluation scripts, configuration files, and result logs to reproduce and extend the experiments.

## Quick summary

- Goal: explore whether sparse sub-networks ("winning tickets") found via iterative magnitude pruning can be re-trained to match the performance of the original dense networks.
- Models: convolutional networks (CNN), LeNet-300-100 experiments, VGG variants, and experiments using BERT for (few-shot) setups.
- Contents: training, pruning, evaluation, plotting and scripts to run experiments and reproduce results.

- **Assignment 2 scope:** LeNet, CNN, VGG experiments  
- **Assignment 3 scope:** BERT experiments

## Repository structure

- `CNN/` — code for convolutional network experiments: data loading (`data.py`), model definitions (`models.py`), pruning (`pruning.py`), training (`trainer.py` / `main.py`), plotting (`plotter.py`), and example run scripts (`run_experiments.sh`). Results are under `CNN/results/`.
- `Lenet-300-100/` — experiments and scripts for the fully-connected LeNet-300-100 family. Includes `train.py`, `run_experiments.py`, `plot_results.py`, and plotting helpers.
- `VGG/` — VGG experiment scripts and training/pruning code.
- `BERT/` — BERT-related scripts for few-shot and LTH experiments, e.g. `few_shot_training.py`, `lottery_ticket_bert.py`, `run_fewshot.sh`, and `run_lth_bert.sh`.
- Top-level files:
	- `pyproject.toml` — project metadata (may be used for tooling).
	- `README.md` — (this file).

## Requirements

This project is Python-based. Exact package lists may differ between subfolders; check per-subfolder requirements.

- Recommended: Python 3.8+ and a recent PyTorch version (for GPU support).
- Check these files for dependencies:
	- `CNN/requirements.txt`
	- `Lenet-300-100/` may include requirements in `requirements.txt` or `pyproject.toml`.

Typical quick setup (macOS / zsh):

```bash
# create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# install requirements for a subproject (example: CNN)
pip install -r CNN/requirements.txt
```

If you prefer conda or poetry, adapt those steps to your workflow.

## Running experiments

Many subfolders include convenience scripts to run the experiments. Examples:

#### CNN experiments (from repo root):

```bash
cd CNN
./run_experiments.sh    # runs a set of experiments (script may accept flags)
```

#### LeNet experiments:

```bash
cd Lenet-300-100
python run_experiments.py
```

#### BERT few-shot / LTH runs in a cluster:

```bash
cd BERT
chmod +x run_lth_bert.sh
./run_lth_bert.sh
chmod +x run_fewshot.sh
./run_fewshot.sh
```

#### How to run locally

##### Run a lottery ticket experiment.
Train and prune BERT on SST-2 or QQP:

```bash
python lottery_ticket_bert.py --task sst2 --sparsity 0.6 --epochs 3 --batch_size 32 --lr 2e-5 --seed 42
```

##### Create Ensemble Tickets

After generating two sparse BERT models (e.g., one trained on SST-2 and another on QQP with the same random seed), you can merge them into a single ensemble ticket that combines their active weights.

```bash
python create_ensemble_tickets.py \
  --ticket_a ./models/lottery_ticket_sst2_sparsity60_seed42.pt \
  --ticket_b ./models/lottery_ticket_qqp_sparsity70_seed42.pt \
  --save_dir ./models
```

##### Few-Shot Fine-Tuning

After training and pruning your sparse BERT models (or creating ensemble tickets), you can perform few-shot fine-tuning to test transferability on another task using limited labeled data.

```bash
python few_shot_training.py \
  --checkpoint ./models/lottery_ticket_sst2_sparsity60_seed42.pt \
  --target qqp \
  --mode headonly \
  --fewshot_size 128 \
  --save_dir ./fewshot_models
```

### 
Notes:
- Many scripts accept configuration files in the same folder (e.g. `configs.txt`, `configs_bert.txt`), or command-line flags. Inspect the script headers or `--help` where available.
- If a script fails due to missing packages, install the subproject-specific requirements or the packages listed in `pyproject.toml`.

## Quick checks / GPU

- A fast GPU availability check (Python):

```bash
python -c "import torch, sys; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
```

- The `CNN/` folder contains a `check_python_gpu.bsub` example for cluster jobs — inspect it if you plan to run on an HPC scheduler.

## Reproducing plots and results

- The `CNN/plotter.py` and `Lenet-300-100/plot_results.py` scripts generate figures from experiment CSV outputs (see `CNN/results/` and `Lenet-300-100/plots/`).
- To reproduce a published figure, locate the corresponding `results_*.csv` file in the relevant `results/` folder and run the plot script in that subfolder.

## Tips for reproducibility

- Seed control: many training scripts accept a random seed parameter — use it to reproduce runs.
- Configs: use the provided config files in each subfolder to match experiment settings.
- Hardware: GPU vs CPU may change runtime and (rarely) numeric differences; prefer GPU for larger runs.

## Tests and verification

Add a minimal smoke test to verify the environment:

```bash
source .venv/bin/activate
python -c "import torch; print('cuda', torch.cuda.is_available()); from pathlib import Path; print('root', Path('.').resolve())"
```

To run unit-like checks, inspect and run small training loops in the subfolders (for example run a single-epoch training in `CNN/main.py` with reduced dataset / batch size).

## Contributing

If you plan to extend or reproduce experiments, please:

1. Create an issue describing the goal or bug.
2. Open a branch and submit a PR with code + small test to validate behavior.

## License & citation

This repository contains research code. If you reuse code or results, please cite the original authors and include a link to this repo in your work.

## Contact

For questions about the code or experiments, open an issue or contact the repository owner / primary authors listed in the paper associated with this project.

---