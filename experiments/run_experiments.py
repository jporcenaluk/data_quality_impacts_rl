"""
This is the main entry point: run the experiments.

Just run the file to do it.
"""

import datetime as dt
import pathlib
from zoneinfo import ZoneInfo
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
import os
import psutil
from typing import Any, Callable, Dict, Iterable, List, Optional
import torch
from itertools import zip_longest
# from stable_baselines3.ppo import PPO as AgentPPO
# from stable_baselines3.sac import SAC as AgentSAC
# from stable_baselines3.ddpg import DDPG as AgentDDPG
# from stable_baselines3.td3 import TD3 as AgentTD3
# from sb3_contrib.ppo_recurrent import RecurrentPPO as AgentRecurrentPPO
# import gymnasium as gym

from run_models.models import Models
from run_models.run_baseline import RunBAS
from run_models.run_rbc import RunRBC
from run_models.run_rl import RunRL

from sbx import SAC as AgentSAC
from sbx import PPO as AgentPPO
from sbx import TD3 as AgentTD3

# helps avoid sharing threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)  # CPU-side ops inside PyTorch
torch.set_num_interop_threads(1)

# supports for this in Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RUN_ID: str = "0060"
TIMESTEP_BUDGET = 5_000_000
LEARNING_STARTS = 10_000
ENV_COUNT = 28
TARGET_EVALS = 50

def get_runs(run_id: str, building_count: int = 1):
    time_now = dt.datetime.now(tz=ZoneInfo("Europe/Dublin")).strftime("%Y-%m-%d_%H-%M-%S")
    run_collection_id = f"{time_now}_{run_id}"
    from run_models.run_metadata import RunMetadata
    from run_models.models import Models
    cwd = str(pathlib.Path(__file__).parent.resolve())

    # metadata_aran = RunMetadata(run_name="sandbox_0011", schema_filepath="/data/datasets/sandbox_hourly/schema.json")
    metadata_pristine = [{model.name: RunMetadata(building_file="Building_01_pristine.csv",
                                    run_name="pristine",
                                    model_type=model.name,
                                    run_collection_id=run_collection_id,
                                    cwd=cwd)} for model in Models]
    metadata_missing = [{model.name: RunMetadata(building_file="Building_01_missing.csv",
                                    run_name="missing",
                                    model_type=model.name,
                                    run_collection_id=run_collection_id,
                                    cwd=cwd)} for model in Models]
    metadata_outliers = [{model.name: RunMetadata(building_file="Building_01_outliers.csv",
                                    run_name="outliers",
                                    model_type=model.name,
                                    run_collection_id=run_collection_id,
                                    cwd=cwd)} for model in Models]

    runs = [metadata_pristine, metadata_missing, metadata_outliers]
    # runs = [metadata_pristine]
    return runs

def process_run(run_metadata):
    """process a run"""
    try:
        bas = run_metadata.get(Models.bas.name)
        rbc = run_metadata.get(Models.rbc.name)
        sac = run_metadata.get(Models.sac.name)
        ppo = run_metadata.get(Models.ppo.name)
        # rppo = run_metadata.get(Models.rppo.name)
        # ddpg = run_metadata.get(Models.ddpg.name)
        td3 = run_metadata.get(Models.td3.name)
        print("Run", run_metadata.keys())

        if bas:
            print("-------Running baseline")
            run = RunBAS(run_metadata=bas)
            run.train()
            run.test()
            print("-------End baseline run")
        elif rbc:
            print("-------Running rbc")
            run = RunRBC(run_metadata=rbc)
            run.train()
            run.test()
            print("-------End rbc run")
        elif sac:
            print("-------Running SAC")
            run = RunRL(run_metadata=sac)
            agent_kwargs = dict(
                ent_coef="auto_0.5",
                learning_starts=LEARNING_STARTS # wait a bit before starting learning
            )
            run.train("MlpPolicy", AgentSAC, env_count=ENV_COUNT, timestep_budget=TIMESTEP_BUDGET, learning_rate=3e-4, gamma=0.99, device="cuda", target_evals=TARGET_EVALS, **agent_kwargs)
            run.test(AgentSAC)
            print("-------End SAC")
        elif ppo:
            print("-------Running PPO")
            run = RunRL(run_metadata=ppo)
            agent_kwargs = dict()
            run.train("MlpPolicy", AgentPPO, env_count=ENV_COUNT, timestep_budget=TIMESTEP_BUDGET, learning_rate=3e-4, gamma=0.99, device="cpu", target_evals=TARGET_EVALS, **agent_kwargs)
            run.test(AgentPPO)
            print("-------End PPO")
        # elif ddpg:
        #     print("-------Running DDPG")
        #     agent_kwargs = dict(
        #         learning_starts=10_000 # wait a bit before starting learning
        #     )
        #     run = RunRL(run_metadata=ddpg)
        #     run.train("MlpPolicy", AgentDDPG, env_count=1, episodes=EPISODES, learning_rate=3e-4, gamma=0.995, device="auto", **agent_kwargs)
        #     run.test(AgentDDPG)
        #     print("-------End DDPG")
        elif td3:
            print("-------Running TD3")
            agent_kwargs = dict(
                learning_starts=LEARNING_STARTS # wait a bit before starting learning
            )
            run = RunRL(run_metadata=td3)
            run.train("MlpPolicy", AgentTD3, env_count=ENV_COUNT, timestep_budget=TIMESTEP_BUDGET, learning_rate=3e-4, gamma=0.99, device="cuda", target_evals=TARGET_EVALS, **agent_kwargs)
            run.test(AgentTD3)
            print("-------End TD3") 

    except Exception as e:
        print("Unable to train model", run_metadata.keys(), e)

def main(all_run_metadata):
    """
    Using available cores, run models in parallel.
    """
    def _is_gpu_rl(rm) -> bool:
        return any(k in rm for k in ("sac", "td3", "ddpg"))
    def _is_cpu_rl(rm) -> bool:
        # extra option makes ppo a string, not p-p-o, which is what happens if it's alone
        return any(k in rm for k in ("ppo",""))
    
    cpu_jobs = [rm for rm in all_run_metadata if _is_cpu_rl(rm)]
    gpu_jobs = [rm for rm in all_run_metadata if _is_gpu_rl(rm)]
    other_jobs = [rm for rm in all_run_metadata if not (_is_cpu_rl(rm) or _is_gpu_rl(rm))]

    # for CPU jobs, good to run in parallel
    def execute_parallel(jobs: list):
        cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        max_workers = min(cpu_count, len(jobs))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_metadata = {
                executor.submit(process_run, rm): rm for rm in jobs
            }

            for future in as_completed(future_to_metadata):
                rm = future_to_metadata[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Run {list(rm.keys())} raised an error {e}")

    def run_dual(
        cpu_items: Iterable[Any],
        gpu_items: Iterable[Any],
        cpu_fn: Callable[[Any], Any],
        gpu_fn: Callable[[Any], Any],
        *,
        cpu_executor: Optional[ProcessPoolExecutor] = None,
        gpu_executor: Optional[ThreadPoolExecutor] = None,
    ) -> Dict[str, List[Any]]:
        """
        Run one CPU-bound job and one GPU-bound job concurrently at all times.
        When one finishes, enqueue the next from that same list.
        When a list is exhausted, the other continues serially.

        Returns dict with 'cpu' and 'gpu' results (in completion order per stream).
        """
        own_cpu = cpu_executor is None
        own_gpu = gpu_executor is None
        if own_cpu:
            cpu_executor = ProcessPoolExecutor(max_workers=1)      # CPU job uses all cores internally
        if own_gpu:
            gpu_executor = ThreadPoolExecutor(max_workers=1)       # Keep GPU in one process/thread

        cpu_iter = iter(cpu_items)
        gpu_iter = iter(gpu_items)
        results = {"cpu": [], "gpu": []}

        try:
            cpu_fut = gpu_fut = None
            try:
                cpu_fut = cpu_executor.submit(cpu_fn, next(cpu_iter))
            except StopIteration:
                cpu_fut = None
            try:
                gpu_fut = gpu_executor.submit(gpu_fn, next(gpu_iter))
            except StopIteration:
                gpu_fut = None

            while cpu_fut or gpu_fut:
                done, _ = wait({f for f in (cpu_fut, gpu_fut) if f}, return_when=FIRST_COMPLETED)
                for f in done:
                    if f is cpu_fut:
                        results["cpu"].append(f.result())  # will raise if cpu_fn errored
                        try:
                            cpu_fut = cpu_executor.submit(cpu_fn, next(cpu_iter))
                        except StopIteration:
                            cpu_fut = None
                    elif f is gpu_fut:
                        results["gpu"].append(f.result())
                        try:
                            gpu_fut = gpu_executor.submit(gpu_fn, next(gpu_iter))
                        except StopIteration:
                            gpu_fut = None
        finally:
            if own_cpu:
                cpu_executor.shutdown(cancel_futures=True)
            if own_gpu:
                gpu_executor.shutdown(cancel_futures=True)

        return results

    out = run_dual(cpu_jobs, gpu_jobs, process_run, process_run)
    print("GPU results", out["gpu"])
    print("CPU results", out["cpu"])
    # run other jobs last
    if len(other_jobs) > 0:
        execute_parallel(other_jobs)


def run_main(run_id: str):

    runs = get_runs(run_id)

    all_run_metadata = [rm for run_group in runs for rm in run_group]

    metadata_pristine = runs[0]
    try:
        log_path = f"{metadata_pristine[0][Models.bas.name].run_collection_path}/process.log"
    except:
        log_path = f"{metadata_pristine[0][Models.ppo.name].run_collection_path}/process.log"
    with open(log_path, "a") as f:
        f.write(f"Process started at {dt.datetime.now()!r}\n")
    main(all_run_metadata)

    with open(log_path, "a") as f:
        f.write(f"Process ended at   {dt.datetime.now()!r}\n")


if __name__ == "__main__":
    run_main(RUN_ID)
