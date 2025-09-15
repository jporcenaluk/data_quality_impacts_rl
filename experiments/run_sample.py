

import pathlib
from run_models.models import Models
from run_models.run_metadata import RunMetadata
from run_models.run_rl import RunRL
from sbx import SAC as AgentSAC
from sbx import PPO as AgentPPO
import datetime as dt
from zoneinfo import ZoneInfo

def run_models(run_id: str):
    time_now = dt.datetime.now(tz=ZoneInfo("Europe/Dublin")).strftime("%Y-%m-%d_%H-%M-%S")
    # time_now = "2025-08-17_13-02-05"
    cwd = str(pathlib.Path(__file__).parent.resolve())
    model_metadata = RunMetadata(building_file="Building_01_pristine.csv", 
                                run_name="pristine",
                                run_collection_id=f"custom_report_{time_now}",
                                model_type=Models.ppo.name,
                                cwd=cwd)
    run = RunRL(run_metadata=model_metadata)
    # run.run_metadata.run_path = 
    run.train(policy="MlpPolicy", agent_cls=AgentPPO, env_count=30, timestep_budget=5000)
    run.test(agent_cls=AgentPPO)

if __name__ == "__main__":
    run_models("test0023")
