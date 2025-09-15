import glob
import os
import pathlib
from typing import Tuple
from experiments.run_models.models import Models
from report.run_info import RunInfo
from experiments.report.run_report import report_actions, report_actions_stack
from experiments.report.run_report import report_learning_curve_ep_mean
from experiments.report.run_report import report_carbon_emissions
from experiments.report.run_report import carbon_emissions_mean
from experiments.report.citylearn_tut import plot_simulation_summary
from pandas import pd

CANONICAL_RUN = "2025-08-30_21-18-41_0058"

def init_run_infos(canonical_run: str) -> Tuple[list, str]:
    """
    Get the run information
    """
    cwd = str(pathlib.Path(__file__).parent.resolve())
    run_folder = f"{cwd}/canonical_runs/{canonical_run}"

    if not pathlib.Path(run_folder).exists():
        raise ValueError(f"Run folder {run_folder} does not exist")
    
    print("-run folder------------")
    print(run_folder)
    run_names = []
    for folder_name in glob.iglob(f"{run_folder}/**"):
        if os.path.isdir(folder_name):
            run_names.append(folder_name)

    print("-run names-------------")
    print(run_names)

    inner_folders: list[str] = []
    for run in run_names:
        for folder in glob.iglob(f"{run}/**"):
            if os.path.isdir(folder):
                inner_folders.append(folder)

    print("-inner folders----------")
    print(inner_folders)

    run_infos = []
    for folder in inner_folders:
        try:
            # This is brittle, but we can use slicing to get the names
            model_name = folder.split("/")[-1]
            model_name_title = model_name.replace("_", " ").upper()
            
            title = folder.split("/")[-2].replace("_", " ").title()
            full_title = f"{model_name_title} {title}"
            
            run_collection_id = folder.split("/")[-3].split("_")[-1]
            
            run_info = RunInfo(id=full_title,
                            path=folder,
                            title=full_title,
                            collection_id=run_collection_id,
                            log_path_relative="tb_logs",
                            algo_type=Models[model_name],
                            building_csv="")
            run_infos.append(run_info)
        except Exception as e:
            print(f"Issue with {folder}", e)
    print("-run infos--------------")
    print(run_infos)

    return run_infos, run_folder

def actions_over_time(run_infos: list[RunInfo], run_folder):
    """
    Viewing actions over time in a chart will reveal what the learning has 'taught' the algorithm to do, given the observations it has seen.

    It might be interesting to compare actions with the carbon intensity, as well as solar generation. There may be trends to view.

    This could be done at the macro level (over a year) or the micro level (a week) to see what plays out.
    """
    for run in run_infos:
        report_actions(run)

    action_stack = [r for r in run_infos if r.algo_type not in [Models.bas]]
    report_actions_stack(action_stack, run_folder)

def learning_curves(run_infos: list[RunInfo], run_folder: str):
    """
    How did the reinforcement learning go? Did any of the models learn faster than others? What does noise have to do with it?

    I need to get the mean rewards over time for each of the RL models and plot them on a single chart. This will show how stable/fast learning is.
    """
    emissions_stack = [r for r in run_infos if r.algo_type in [Models.ppo, Models.sac, Models.ddpg, Models.td3]]
    report_learning_curve_ep_mean(emissions_stack, run_folder)

def citylearn_tut(run_infos: list[RunInfo], run_folder: str):
    """
    The citylearn tutorials have reportst that are generated; let's see what those look like.
    """
    plot_simulation_summary(run_infos=run_infos, run_folder=run_folder)

def plot_soc(run_infos: list[RunInfo], run_folder: str):
    """
    plot soc
    """
    files: list[str] = os.listdir(run_folder)
    obs = [f in files if "building_obs" in f]
    for f in obs:
        df = pd.read_csv(f)
        

if __name__ == "__main__":
    run_infos, run_folder = init_run_infos(CANONICAL_RUN)
    actions_over_time(run_infos=run_infos, run_folder=run_folder)
    learning_curves(run_infos=run_infos, run_folder=run_folder)
    report_carbon_emissions(run_infos=run_infos, run_folder=run_folder)
    carbon_emissions_mean(run_infos=run_infos, run_folder=run_folder)
    plot_simulation_summary(run_infos=run_infos, run_folder=run_folder)