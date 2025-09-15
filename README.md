###

This capstone project uses Reinforcement Learning to train how to use grid-connected batteries more effectively. It contains a thesis and experiments, as well as data.

## Environment Setup

Requires miniconda to be installed; if you don't have miniconda installed you can run:

```bash
source quickstart-lambda-labs.sh
```

Otherwise run:

```bash
source quickstart.sh
```

## Run tests

```
python3 -m pytest
```

### Folders

`experiments` - holds all the code and output of experiments
`glossary` - a smattering of terms that may be used (incomplete, could be deleted)
`journal` - memos about the project as it unfolded
`literature_review` - lit review (precursor to the capstone project as a whole)
`eda`- graveyard of exploratory notebooks
`thesis` - the thesis
`thesis_template` - the original template as provided by University of Galway to write the thesis

### Running Experiments

Before running any experiments, look at the `run_sample.py`, `run_experiments.py`, and `run_reports.py` files under the `experiments` folder. Change any hard-coded constants at the top of the file; these are intended to be run-specific.

To run a sample experiment, execute:

```bash
cd experiments
python3 run_sample.py
```

To run all experiments, execute:

```bash
cd experiments
tmux new -s run_experiment_sesh
conda activate citylearn
python3 run_experiments.py > run_experiment.log
```

And if you want to stop the experiments:

```bash
tmux attach -t run_experiment_sesh
```

To generate reports, execute:

```bash
cd experiments
python3 run_reports.py
```

### Seeing experimental output

Experiments will be written to the `experiments/learn` directory.

Moving them into the `experiments/canonical_runs` directory is how runs are cherry-picked. This is the directory from which `run_reports.py` looks at (although you still have to specify which folder you want to create reports for). Report outputs - such as .png files - are written to this directory.