import math
from typing import Any, List, Mapping
from citylearn.reward_function import RewardFunction
from typing import Any, List, Mapping

import numpy as np

class Normalise:
    """
    Ensures rewards are scaled appropriately to
    rewards received in the past.

    Initialize with a small number so it does not divide by zero, and go from there.
    """
    max: float = 1.0

    def update(self, current_value: float) -> bool:
        """
        Update the maximum value

        It is assumed any increase greater than 1 in increase is an outlier,
        as values should move slowly, so value changes greater than 1
        are ignored.

        :current_value: current value
        :returns: True if updated, False if not updated
        """
        # if the increase in max value is greater than 50% larger or smaller, ignore
        current_value_abs = abs(current_value)
        if current_value_abs > (self.max * 1.5):
            # print("oops, too big of a change", current_value, self.max)
            return False
        self.max = max(self.max, current_value_abs)
        return True

class CarbonRewardV1(RewardFunction):
    """
    Rewards lowering emissions. Returns reward between -1 and 1.
    """
    net_electricity_consumption_norm = Normalise()
    carbon_intensity_norm = Normalise()
    reward_norm = Normalise()

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, float | int]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        net_electricity_consumptions = [o['net_electricity_consumption'] for o in observations]
        # update norms
        [self.net_electricity_consumption_norm.update(o) for o in net_electricity_consumptions]

        #### Carbon intensity reward calculation
        # carbon intensity should never be below zero... and we'll make sure of it
        # unless someday we make a carbon negative grid. Can you imagine!?!
        # carbon intensity is the same across all dwellings (all pull from grid), so setting to first environment
        carbon_intensity = max(observations[0]['carbon_intensity'], 0)
        self.carbon_intensity_norm.update(carbon_intensity)
        
        # get normalized electricity
        net_electricity_norms = [n / self.net_electricity_consumption_norm.max for n in net_electricity_consumptions]
        # get normalized carbon emissions
        carbon_intensity_norm = carbon_intensity / self.net_electricity_consumption_norm.max

        ## Baseline:             -(1 * 0 * 0) = 0
        ## High Net Electricity: -(1 * 2 * 1) = -2
        ## High Carbon Emission: -(1 * 1 * 2) = -2
        ## Low Carbon Emissions: -(1 * 1 * 0.5) = -0.5
        ## Low Net Electricity:  -(1 * 0.5 * 1) = -0.5
        ## Neg Net Electricity:  -(1 * -1 * 1) = 1
        carbon_rewards = [-(1 * n * carbon_intensity_norm) for n in net_electricity_norms]

        if self.central_agent:
            reward = [sum(carbon_rewards)]
        else:
            reward = carbon_rewards
        return reward


class CarbonRewardV2(RewardFunction):
    """
    Rewards lowering emissions. Adds in limiter for outliers;
    assumes any increase greater than 1 is an outlier and returns a 0 reward
    to protect against learning instability.
    """

    net_electricity_consumption_norm = Normalise()
    carbon_intensity_norm = Normalise()

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, float | int]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        net_electricity_consumptions = [o['net_electricity_consumption'] for o in observations]
        # update norms
        is_electricity_norm_updated = [self.net_electricity_consumption_norm.update(o) for o in net_electricity_consumptions]

        #### Carbon intensity reward calculation
        # carbon intensity should never be below zero... and we'll make sure of it
        # unless someday we make a carbon negative grid. Can you imagine!?!
        # carbon intensity is the same across all dwellings (all pull from grid), so setting to first environment
        carbon_intensity = max(observations[0]['carbon_intensity'], 0)
        is_carbon_intensity_norm_updated = self.carbon_intensity_norm.update(carbon_intensity)

        # end early if updates didn't occur (outliers found)
        is_outlier_detected = is_carbon_intensity_norm_updated == False or not all(is_electricity_norm_updated)
        if self.central_agent and is_outlier_detected:
            return [0.0] # return a neutral reward
        elif is_outlier_detected:
            return [0.0] * len(observations)
        
        # get normalized electricity
        net_electricity_norms = [n / self.net_electricity_consumption_norm.max for n in net_electricity_consumptions]
        # get normalized carbon emissions
        carbon_intensity_norm = carbon_intensity / self.net_electricity_consumption_norm.max

        ## Baseline:             -(1 * 0 * 0) = 0
        ## High Net Electricity: -(1 * 2 * 1) = -2
        ## High Carbon Emission: -(1 * 1 * 2) = -2
        ## Low Carbon Emissions: -(1 * 1 * 0.5) = -0.5
        ## Low Net Electricity:  -(1 * 0.5 * 1) = -0.5
        ## Neg Net Electricity:  -(1 * -1 * 1) = 1
        carbon_rewards = [-(1 * n * carbon_intensity_norm) for n in net_electricity_norms]

        if self.central_agent:
            reward = [sum(carbon_rewards)]
        else:
            reward = carbon_rewards
        return reward
    

class CarbonRewardV3(RewardFunction):
    """
    Rewards lowering emissions. Assumes normalisation wrapper.
    """
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, float | int]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        carbon_intensity = max(observations[0]['carbon_intensity'], 0.0)
        # only penalize grid imports; no credit for exports
        net_kwh = [o['net_electricity_consumption'] for o in observations]
        import_only = [max(n, 0.0) for n in net_kwh]
        r = [-carbon_intensity * n for n in import_only] # this kind of works like a ReLU in a certain fashion
        return [sum(r)] if self.central_agent else r

class CarbonRewardV4(RewardFunction):
    def __init__(self, env_metadata: dict[str, Any]):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        """

        super().__init__(env_metadata)

    def calculate(
        self, observations: list[dict[str, int | float]]
    ) -> list[float]:
        r"""Returns reward for most recent action.

        The reward is designed to minimize carbon emissions.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        It encourages net-zero energy use by penalizing grid load satisfaction
        when there is energy in the battery as well as penalizing
        net export when the battery is not fully charged through the penalty
        term. There is neither penalty nor reward when the battery
        is fully charged during net export to the grid. Whereas, when the
        battery is charged to capacity and there is net import from the
        grid the penalty is maximized.

        This is based on the reward from the citylearn tutorial:
        https://github.com/intelligent-environments-lab/CityLearn/blob/8110779b125a8d5e44cb15234990c709495e57e4/examples/tutorial.ipynb
        
        Parameters
        ----------
        observations: list[dict[str, int | float]]
            List of all building observations at current
            :py:attr:`citylearn.citylearn.CityLearnEnv.time_step`
            that are got from calling
            :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: list[float]
            Reward for transition to current timestep.
        """

        reward_list = []

        for o, m in zip(observations, self.env_metadata['buildings']):
            cost = o['net_electricity_consumption']*o['carbon_intensity']
            battery_soc = o['electrical_storage_soc']
            penalty = -(1.0 + np.sign(cost)*battery_soc)
            reward = penalty*abs(cost)
            reward_list.append(reward)

        reward = [float(sum(reward_list))]

        return reward

class CarbonRewardV5(RewardFunction):
    """
    Rewards lowering emissions. Assumes normalisation wrapper.
    """
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)
        self.soc0 = None
        self.prev_soc = None
        self.alpha_throughput = 0.08
        self.alpha_anchor = 0.2
        self.anchor_soc = None
        # self.prev_soc = None

    # def __post_init__(self):
        # self.T = 
    # def _soc(self, observations):
    #     return [o["electrical_storage_soc"] for o in observations]

    def reset(self):
        self.soc0 = None
        self.prev_soc = None

    def calculate(self, observations: List[Mapping[str, float | int]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        carbon_intensity = max(observations[0]["carbon_intensity"], 0.0)
        electricity_imports = [max(o["net_electricity_consumption"], 0.0) for o in observations]
        socs = [o["electrical_storage_soc"] for o in observations]
        self.anchor_socs = [0.5]*len(socs)
        if self.soc0 is None:
            self.soc0 = socs[:]
        if self.prev_soc is None:
            self.prev_soc = socs[:]
        
        # penalize emissions
        # penalize throughput (e.g. 'dithering' and doing nothing)
        # penalize going too far away from 50% battery (to avoid only discharge)
        r_per_b = [(-carbon_intensity * electricity_import) -
                   (self.alpha_throughput * abs(soc - prev_soc) -
                   (self.alpha_anchor * ((soc - anchor_soc) ** 2))
                 )
                 for electricity_import, soc, prev_soc, anchor_soc in zip(electricity_imports, socs, self.prev_soc, self.anchor_socs)]

        # return mean rather than sum (to stay agnostic to number of buildings, and to reward all buildings not just one)
        return [float(np.mean(r_per_b))] if self.central_agent else r_per_b


class CarbonRewardV6(RewardFunction):
    """
    Rewards lowering emissions, heavily penalizes via exponent. Assumes normalisation wrapper.
    """
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, float | int]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        carbon_intensity = max(observations[0]['carbon_intensity'], 0.0)
        # adds one before exponential to avoid making reward too small (vanishing reward)
        carbon_intensity_exp = (1 + carbon_intensity) ** 10
        # only penalize grid imports; no credit for exports
        net_kwh = [o['net_electricity_consumption'] for o in observations]
        import_only = [max(n, 0.0) for n in net_kwh]
        r = [-carbon_intensity_exp * n for n in import_only] # this kind of works like a ReLU in a certain fashion
        return [sum(r)] if self.central_agent else r



class CarbonRewardV7(RewardFunction):
    """
    Rewards lowering emissions. Assumes normalisation wrapper.
    """
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)
        self.soc0 = None
        self.prev_soc = None

    # def __post_init__(self):
        # self.T = 
    # def _soc(self, observations):
    #     return [o["electrical_storage_soc"] for o in observations]

    def reset(self):
        self.soc0 = None
        self.prev_soc = None

    def calculate(self, observations: List[Mapping[str, float | int]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """
        ci = max(observations[0]["carbon_intensity"], 0.0)           # [~0.1, 0.4]
        imports = [o["net_electricity_consumption"] for o in observations]  # kWh (+ import, - export)
        socs = [o["electrical_storage_soc"] for o in observations]
        if self.prev_soc is None: self.prev_soc = socs[:]

        # knobs you’ll tune
        beta_throughput = 0.5   # penalize dithering (small)
        alpha_anchor    = 0.05  # softly pull to 50% (small)
        k_nudge         = 6.0   # nudge discharge when CI is high, charge when low (small)

        ci_ref = 0.16           # mid of your stated range
        r_per_b = []
        for e, soc, soc_prev in zip(imports, socs, self.prev_soc):
            d_soc = soc - soc_prev
            # 1) Primary objective: reduce emissions (use built-in emission if available)
            r = -(ci * e)  # or: r = -o['net_electricity_consumption_emission']
            # 2) Discourage pointless cycling and extremes
            r += -beta_throughput * abs(d_soc)
            r += -alpha_anchor * (soc - 0.5)**2
            # 3) Tiny, direction-aware nudge: discharge when CI>ref, charge when CI<ref
            r += k_nudge * (ci - ci_ref) * (-d_soc)
            r_per_b.append(r)

        self.prev_soc = socs[:]
        return [float(np.mean(r_per_b))] if self.central_agent else r_per_b
