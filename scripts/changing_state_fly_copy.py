import numpy as np
from tqdm import trange
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from flygym.simulation import SingleFlySimulation, Fly
from flygym.examples.cpg_controller import CPGNetwork
from flygym.preprogrammed import get_cpg_biases

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from dm_control import mjcf
from dm_control.utils import transformations
from gymnasium import spaces
from gymnasium.core import ObsType
from scipy.spatial.transform import Rotation as R

import flygym.preprogrammed as preprogrammed
import flygym.state as state
import flygym.util as util
import flygym.vision as vision
from flygym.arena import BaseArena
from flygym.util import get_data_path

from .preprogrammed_steps import PreprogrammedSteps


if TYPE_CHECKING:
    from flygym.simulation import Simulation



_tripod_phase_biases = get_cpg_biases("tripod")
_tripod_coupling_weights = (_tripod_phase_biases > 0) * 10
_default_correction_vectors = {
    "F": np.array([0, 0, 0, -0.02, 0, 0.016, 0]),
    "M": np.array([-0.015, 0, 0, 0.004, 0, 0.01, -0.008]),
    "H": np.array([0, 0, 0, -0.01, 0, 0.005, 0]),
}
_default_correction_rates = {"retraction": (500, 1000 / 3), "stumbling": (2000, 500)}
_contact_sensor_placements = tuple(
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
)
variant = "courtship"

class ChangingStateFly(Fly):
    def __init__(
            self, 
            timestep,
            xml_variant=variant,
            preprogrammed_steps=None,
            intrinsic_freqs=np.ones(6) * 12,
            intrinsic_amps=np.ones(6) * 1,
            phase_biases=_tripod_phase_biases,
            coupling_weights=_tripod_coupling_weights,
            convergence_coefs=np.ones(6) * 20,
            init_phases=None,
            init_magnitudes=None,
            stumble_segments=("Tibia", "Tarsus1", "Tarsus2"),
            stumbling_force_threshold=-1,
            correction_vectors=_default_correction_vectors,
            correction_rates=_default_correction_rates,
            amplitude_range=(-0.5, 1.5),
            draw_corrections=False,
            contact_sensor_placements=_contact_sensor_placements,
            seed=0,
            desired_distance=0.2, 
            obj_threshold=0.15, 
            decision_interval=0.05,
            arousal_state=1,
            wings_state=0,
            crab_state=0,
            **kwargs,
        ):
        # Initialize core NMF simulation
        super().__init__(contact_sensor_placements=contact_sensor_placements, xml_variant=xml_variant, **kwargs, enable_vision=True)

        if preprogrammed_steps is None:
            preprogrammed_steps = PreprogrammedSteps()

        self.preprogrammed_steps = preprogrammed_steps
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.phase_biases = phase_biases
        self.coupling_weights = coupling_weights
        self.convergence_coefs = convergence_coefs
        self.stumble_segments = stumble_segments
        self.stumbling_force_threshold = stumbling_force_threshold
        self.correction_vectors = correction_vectors
        self.correction_rates = correction_rates
        self.amplitude_range = amplitude_range
        self.draw_corrections = draw_corrections
        self._set_joints_stiffness_and_damping()

        # Define action and observation spaces
        self.action_space = spaces.Box(*amplitude_range, shape=(2,))

        # Initialize CPG network
        self.cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            seed=seed,
        )
        self.cpg_network.reset(init_phases, init_magnitudes)

        # Initialize variables tracking the correction amount
        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)

        # Find stumbling sensors
        self.stumbling_sensors = self._find_stumbling_sensor_indices()

        # initialize chasing fly parameters
        self.desired_distance = desired_distance
        self.obj_threshold = obj_threshold
        self.decision_interval = decision_interval
        self.num_substeps = int(self.decision_interval / self.timestep)
        self.arousal_state = arousal_state
        self.wings_state = wings_state
        self.crab_state = crab_state
        self.visual_inputs_hist = []
        self.coms = np.empty((self.retina.num_ommatidia_per_eye, 2))
        self.timesteps_at_desired_distance = 0

        for i in range(self.retina.num_ommatidia_per_eye):
            mask = self.retina.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)

    #----------------------------- CHASING FLY --------------------------------
    @property
    def timestep(self):
        return self.cpg_network.timestep

    def _find_stumbling_sensor_indices(self):
        stumbling_sensors = {leg: [] for leg in self.preprogrammed_steps.legs}
        for i, sensor_name in enumerate(self.contact_sensor_placements):
            leg = sensor_name.split("/")[1][:2]  # sensor_name: e.g. "Animat/LFTarsus1"
            segment = sensor_name.split("/")[1][2:]
            if segment in self.stumble_segments:
                stumbling_sensors[leg].append(i)
        stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}
        if any(
            v.size != len(self.stumble_segments) for v in stumbling_sensors.values()
        ):
            raise RuntimeError(
                "Contact detection must be enabled for all tibia, tarsus1, and tarsus2 "
                "segments for stumbling detection."
            )
        return stumbling_sensors

    def _retraction_rule_find_leg(self, obs):
        """Returns the index of the leg that needs to be retracted, or None
        if none applies."""
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
        else:
            leg_to_correct_retraction = None
        return leg_to_correct_retraction

    def _stumbling_rule_check_condition(self, obs, leg):
        """Return True if the leg is stumbling, False otherwise."""
        # update stumbling correction amounts
        contact_forces = obs["contact_forces"][self.stumbling_sensors[leg], :]
        fly_orientation = obs["fly_orientation"]
        # force projection should be negative if against fly orientation
        force_proj = np.dot(contact_forces, fly_orientation)
        return (force_proj < self.stumbling_force_threshold).any()

    def _get_net_correction(self, retraction_correction, stumbling_correction):
        """Retraction correction has priority."""
        if retraction_correction > 0:
            return retraction_correction
        return stumbling_correction

    def _update_correction_amount(
        self, condition, curr_amount, correction_rates, viz_segment
    ):
        """Update correction amount and color code leg segment.

        Parameters
        ----------
        condition : bool
            Whether the correction condition is met.
        curr_amount : float
            Current correction amount.
        correction_rates : Tuple[float, float]
            Correction rates for increment and decrement.
        viz_segment : str
            Name of the segment to color code. If None, no color coding is
            done.

        Returns
        -------
        float
            Updated correction amount.
        """
        if condition:  # lift leg
            increment = correction_rates[0] * self.timestep
            new_amount = curr_amount + increment
            color = (0, 1, 0, 1)
        else:  # condition no longer met, lower leg
            decrement = correction_rates[1] * self.timestep
            new_amount = max(0, curr_amount - decrement)
            color = (1, 0, 0, 1)
        if viz_segment is not None:
            self.change_segment_color(viz_segment, color)
        return new_amount

    def reset(self, sim, seed=None, init_phases=None, init_magnitudes=None, **kwargs):
        obs, info = super().reset(sim, seed=seed, **kwargs)
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
        self.cpg_network.reset(init_phases, init_magnitudes)
        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)
        return obs, info

    
    #----------------------------- CHASING FLY --------------------------------
    def process_visual_observation(self, vision_input):
        features = np.zeros((2, 3))
        dist = np.zeros(2)

        for i, ommatidia_readings in enumerate(vision_input):
            is_obj = ommatidia_readings.max(axis=1) < self.obj_threshold
            is_obj_coords = self.coms[is_obj]

            if is_obj_coords.shape[0] > 0:
                features[i, :2] = is_obj_coords.mean(axis=0)
                # Estimate the distance to the other fly by counting the number of "pixels" perceiving the other fly
                dist[i] = is_obj.sum()

            features[i, 2] = is_obj_coords.shape[0]

        features[:, 0] /= self.retina.nrows  # normalize y_center
        features[:, 1] /= self.retina.ncols  # normalize x_center
        features[:, 2] /= self.retina.num_ommatidia_per_eye  # normalize area
        # Normalize distance (you may need to adjust this depending on the scale of your simulation)
        dist /= self.retina.num_ommatidia_per_eye  # normalize distance by the total number of "pixels"
        return features.ravel().astype("float32"), dist.sum()

    def calc_ipsilateral_speed(self, deviation, is_found):
        if not is_found:
            return 1.0
        else:
            return np.clip(1 - deviation * 3, 0.4, 1.2)

    def calc_walking_speed(self, proximity):
        """
        Calculates the walking speed based on the distance to the other fly.

        Parameters:
        - proximity (float): The proximity to the other fly.

        Returns:
        - speed (float): The calculated walking speed.
        """
        if proximity == 0:
            speed = 1.0
        else:
            speed = self.desired_distance / proximity
        return speed
    
    #----------------------------- CHANGING STATE FLY --------------------------------
    def get_chasing_action(self, obs):
        """
        Get the action to take when chasing the other fly.

        Parameters:
        - obs (np.ndarray): The observation.

        Returns:
        - action (np.ndarray): The action.
        """
        visual_features, proximity = self.process_visual_observation(obs["vision"])
        left_deviation = 1 - visual_features[1]
        right_deviation = visual_features[4]
        left_found = visual_features[2] > 0.005
        right_found = visual_features[5] > 0.005

        if not left_found:
            left_deviation = np.nan

        if not right_found:
            right_deviation = np.nan

        # Calculate fly 0 action
        fly_action = np.array(
            [
                self.calc_ipsilateral_speed(left_deviation, left_found),
                self.calc_ipsilateral_speed(right_deviation, right_found),
            ]
        )

        walking_speed = self.calc_walking_speed(proximity)
        fly_action *= walking_speed

        # TODO: detect long stops and switch gait / wings vibrations

        return fly_action, proximity

    def get_random_action(self, curr_time):
        """
        Get a random action as hybrid turning fly.

        Returns:
        - action (np.ndarray): The random action.
        """
        proximity = None
        if curr_time < 1:
            action = np.array([1.2, 0.2])
        else:
            action = np.array([0.2, 1.2])

        return action, proximity
    '''
    def get_crab_action(self) :
        proximity = None
        action = np.array([0.2, 1.2])
        return action, proximity
    '''
    def get_action(self, obs, curr_time):
        """
        Get the action to take based on the arousal state: 
        if arousal = 1, the fly will walk towards the other fly using visual processing;
        if arousal = 0, the fly will keeps random turning.

        Parameters:
        - obs (np.ndarray): The observation.

        Returns:
        - action (np.ndarray): The action.
        """

        if self.arousal_state == 1:
            return self.get_chasing_action(obs)
        #elif self.crab_state == 1:
        #    return self.get_crab_action()
        else:
            return self.get_random_action(curr_time)
        
        
        


    def update_state(self, obs):
        """
        Update the wings state based on the observation.
        Update the arousal state based on the observation.

        Parameters:
        - obs (np.ndarray): The observation.
        """
        # TODO
        visual_features, proximity = self.process_visual_observation(obs["vision"])

        if self.arousal_state == 0 and proximity < self.desired_distance*3:
            self.arousal_state = 1

        if proximity < self.desired_distance :
            self.timesteps_at_desired_distance = 0
            self.wings_state = 0
            self.crab_state = 0
        elif proximity >= self.desired_distance and self.timesteps_at_desired_distance < 10:
            self.timesteps_at_desired_distance += 1
        elif proximity >= self.desired_distance and self.timesteps_at_desired_distance >= 10:
            self.wings_state = 1
            self.crab_state = 1
            self.arousal_state = 0  #To stop the chasing action
    
    def pre_step(self, action, sim):
        """Step the simulation forward one timestep.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (2,) containing descending signal encoding
            turning.
        """
        physics = sim.physics          

        # update CPG parameters
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        # get current observation
        obs = super().get_observation(sim)

        # Retraction rule: is any leg stuck in a gap and needing to be retracted?
        leg_to_correct_retraction = self._retraction_rule_find_leg(obs)

        self.cpg_network.step()

        joints_angles = []
        adhesion_onoff = []

        # Get legs joint angles
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # update retraction correction amounts
            self.retraction_correction[i] = self._update_correction_amount(
                condition=(i == leg_to_correct_retraction),
                curr_amount=self.retraction_correction[i],
                correction_rates=self.correction_rates["retraction"],
                viz_segment=f"{leg}Tibia" if self.draw_corrections else None,
            )
            # update stumbling correction amounts
            self.stumbling_correction[i] = self._update_correction_amount(
                condition=self._stumbling_rule_check_condition(obs, leg),
                curr_amount=self.stumbling_correction[i],
                correction_rates=self.correction_rates["stumbling"],
                viz_segment=f"{leg}Femur" if self.draw_corrections else None,
            )
            # get net correction amount
            net_correction = self._get_net_correction(
                self.retraction_correction[i], self.stumbling_correction[i]
            )

            # get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            my_joints_angles += net_correction * self.correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )
            adhesion_onoff.append(my_adhesion_onoff)

        # Get wings joint   angles
        # for i, wing in enumerate(self.preprogrammed_steps.wings):
        my_joints_angles = self.get_wings_joint_angles(self.cpg_network.curr_phases[0])
        joints_angles.append(my_joints_angles)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }
        '''
        joint_action = action["joints"]
        if self.crab_state == 1 : 
            for joint in range(len(joint_action)):
                if joint == 4 :
                    joint_action[joint] = 0.3 #LF
                elif joint == 11 : 
                    joint_action[joint] = 0.6 #LM
                elif joint == 18 : 
                    joint_action[joint] = 1 #LH
                elif joint == 25 :
                    joint_action[joint] = 0.3 #RF
                #elif joint <= 34 : 
                elif joint == 32 : 
                    joint_action[joint] = 0.6 # RM
                elif joint == 40 : 
                    joint_action[joint] = 1 #RH  
        '''
        # TODO: update wings state
        self.update_state(obs)

        return super().pre_step(action, sim)
    
    def get_wings_joint_angles(self, phase):
        """Get joint angles for both wings.

        Parameters
        ----------
        wing : str
            Wing name.

        Returns
        -------
        np.ndarray
            Joint angles of the wing. The shape of the array is (6,)
        """
        if self.wings_state == 0:
            return np.array([0, 0, 0, 0, 0, 0])
        elif self.wings_state == 1:
            return self.preprogrammed_steps.get_wing_angles(phase)
    

    def _set_joints_stiffness_and_damping(self):
        for joint in self.model.find_all("joint"):
            if joint.name in self.actuated_joints:
                joint.stiffness = self.joint_stiffness
                joint.damping = self.joint_damping
            else:
                joint.stiffness = self.non_actuated_joint_stiffness
                joint.damping = self.non_actuated_joint_damping