import numpy as np
from gymnasium import spaces
from flygym.simulation import Fly
from flygym.examples.cpg_controller import CPGNetwork
from flygym.preprogrammed import get_cpg_biases
from typing import TYPE_CHECKING
from gymnasium import spaces
from .preprogrammed_steps import PreprogrammedSteps
from flygym.preprogrammed import all_leg_dofs

# Define some constants as in the previous classes
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

# Define variables specific for the ChangingStateFly class
wings_dofs = ["joint_LWing_roll", "joint_LWing_yaw", "joint_LWing", "joint_RWing_roll", "joint_RWing_yaw", "joint_RWing"]
all_dofs = all_leg_dofs + wings_dofs
variant = "courtship"
threshold_switch = 1000
threshold_wings_open = 5000
speed_thresh = 0.2
desired_distance = 0.01
threshold_wings_closed = 2000

class ChangingStateFly(Fly):
    def __init__(
            self, 
            timestep,
            xml_variant=variant,
            actuated_joints=all_dofs,
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
            desired_distance=desired_distance, 
            obj_threshold=0.15, 
            decision_interval=0.05,
            arousal_state=0,
            wings_state=0,
            time_crab=0,
            crab_state=0,
            **kwargs,
        ):
        # Initialize core NMF simulation
        super().__init__(contact_sensor_placements=contact_sensor_placements, xml_variant=xml_variant, actuated_joints=actuated_joints, **kwargs, enable_vision=True)

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
        self.time_crab = time_crab
        self.visual_inputs_hist = []
        self.coms = np.empty((self.retina.num_ommatidia_per_eye, 2))
        self.timesteps_at_desired_distance = 0
        self.timesteps_wings_open = 0
        self.last_open_wing = None

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

    
    #----------------------------- VISUAL TAXIS FLY --------------------------------
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

        return fly_action, proximity

    def get_random_action_2(self, curr_time, obs):
        """
        Get a random action as hybrid turning fly.

        Returns:
        - action (np.ndarray): The random action.
        """
        _, proximity = self.process_visual_observation(obs["vision"])
        if curr_time > 1:
            action = np.array([1.2, 0.2])
        else:
            action = np.array([0.2, 1.2])

        return action, proximity
    
    import numpy as np

    def get_random_action(self, curr_time, obs):
        """
        Get a random action as hybrid turning fly.

        Parameters:
        - curr_time (float): The current time in the simulation.
        - obs (np.ndarray): The observation.

        Returns:
        - action (np.ndarray): The random action.
        - proximity (float): The proximity value from the observation.
        """
        _, proximity = self.process_visual_observation(obs["vision"])

        # Initialize or update the direction and period
        if not hasattr(self, 'current_direction'):
            self.current_direction = 'right'
            self.time_since_switch = 0
            self.current_period = np.random.randint(100, 200)  # Random period between 100 and 200 timesteps

        # Update the time since the last switch
        self.time_since_switch += 1

        # Switch direction if the current period has expired
        if self.time_since_switch >= self.current_period:
            self.current_direction = 'left' if self.current_direction == 'right' else 'right'
            self.time_since_switch = 0
            self.current_period = np.random.randint(100, 200)  # Generate a new random period

        # Set the action based on the current direction
        if self.current_direction == 'right':
            action = np.array([1.2, 0.2])
        else:
            action = np.array([0.2, 1.2])

        return action, proximity

    
    def get_crab_action(self, obs) :
        _, proximity = self.process_visual_observation(obs["vision"])
        if self.crab_state == 1:
            action = np.array([0, 1.3])
        if self.crab_state == 2: 
            action = np.array([2, 0])
        if self.crab_state == 3: 
            action = np.array([0, 0])  
        if self.crab_state == 4: 
            action = np.array([1, 1]) 
        return action, proximity
    
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

        if self.arousal_state == 1 and self.crab_state == 0:
            return self.get_chasing_action(obs)
        elif self.arousal_state == 1 and self.crab_state != 0:
            return self.get_crab_action(obs)
        else:
            return self.get_random_action(curr_time, obs)
        

    def update_state(self, obs):
        """
        Update the wings state based on the observation.
        Update the arousal state based on the observation.

        Parameters:
        - obs (np.ndarray): The observation.
        """
        visual_features, proximity = self.process_visual_observation(obs["vision"])
        speed = self.calc_walking_speed(proximity)

        # Update arousal state if the other fly is close
        if self.arousal_state == 0 and proximity < self.desired_distance/2:
            self.arousal_state = 1

        # Switch state if the fly stays at the desired distance for a certain number of timesteps
        if speed > speed_thresh and self.crab_state == 0:
            self.timesteps_at_desired_distance = 0
            self.wings_state = 0
            self.crab_state = 0
        elif speed < speed_thresh and self.timesteps_at_desired_distance < threshold_switch:
            self.timesteps_at_desired_distance += 1
        elif speed < speed_thresh and self.timesteps_at_desired_distance >= threshold_switch and self.crab_state == 0 :
            self.crab_state = 1
        elif self.crab_state == 1 and self.time_crab < 3000:
            self.time_crab += 1
        elif (self.crab_state == 1 and self.time_crab <4500 ) or (self.crab_state == 4 and self.time_crab < 4500 ) :
            self.time_crab += 1
            self.crab_state = 4
        elif self.time_crab < 7000:
            self.crab_state = 2
            self.time_crab += 1
            # self.wings_state = 1       
        elif self.wings_state==0:
            self.crab_state = 3
            self.wings_state = 1

        # Update wings and crabe state
        if self.wings_state == 1:
            self.timesteps_wings_open += 1
            if self.timesteps_wings_open >= threshold_wings_open:
                self.wings_state = 2
                self.timesteps_wings_open = 0

                left_deviation = 1 - visual_features[1]
                right_deviation = visual_features[4]

                if left_deviation > right_deviation:
                    self.last_open_wing = 'L'
                else:
                    self.last_open_wing = 'R'

        if self.wings_state == 2: 
            self.timesteps_wings_open += 1
            if self.timesteps_wings_open >= threshold_wings_closed:
                self.wings_state = 1
                self.timesteps_wings_open = 0

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
        my_joints_angles = self.get_wings_joint_angles(self.last_open_wing, obs)
        joints_angles.append(my_joints_angles)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }

        # Update fly state
        self.update_state(obs)

        return super().pre_step(action, sim)
    
    def get_joint_angles_crabe_walk(self, joint_angles, leg_to_correct):
        """
        Get the joint angles for the crabe walk from the normal walk joint angles.

        Parameters:
        - joint_angles (np.ndarray): The joint angles.

        Returns:
        - joint_angles (np.ndarray): The joint angles for the crabe walk.
        """
        joint_angles_prev = joint_angles.copy()
        
        # TODO: switch the right joint angles
        # for leg in range(6):
        #     for dof in range(7):
        #         if dof == 0: # Coxa pitch
        #             joint_angles[leg * 7 + dof] = joint_angles_prev[leg * 7 + dof]
        #         elif dof == 1: # Coxa roll
        #             joint_angles[leg * 7 + dof] = 0.6
        #         elif dof == 2: # Coxa yaw
        #             joint_angles[leg * 7 + dof] = 1
        #         elif dof == 3: # Femur pitch
        #             joint_angles[leg * 7 + dof] = 0.3
        #         elif dof == 4: # Femur roll
        #             joint_angles[leg * 7 + dof] = 0.6
        #         elif dof == 5: # Tibia pitch
        #             joint_angles[leg * 7 + dof] = 1
        #         elif dof == 6: # Tarsus roll
        #             joint_angles[leg * 7 + dof] = 0.3

        if leg_to_correct == 'L':
            for leg in range(3):
                for dof in range(7):
                    joint_angles[leg * 7 + dof] = -joint_angles_prev[leg * 7 + dof]
        elif leg_to_correct == 'R':
            for leg in range(3,6):
                for dof in range(7):
                    joint_angles[leg * 7 + dof] = joint_angles_prev[leg * 7 + dof]

        return joint_angles

    
    def get_wings_joint_angles(self, wing_to_open, obs):
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
        if self.wings_state == 0 or self.wings_state == 2:
            return np.array([0, 0, 0, 0, 0, 0])
        elif self.wings_state == 1:
            if wing_to_open == 'L':
                return np.array([-1.2, 0, 0, 0, 0, 0])
            elif wing_to_open == 'R':
                return np.array([0, 0, 0, 1.2, 0, 0])
            else:
                visual_features, proximity = self.process_visual_observation(obs["vision"])
                left_deviation = 1 - visual_features[1]
                right_deviation = visual_features[4]
                if left_deviation < right_deviation:
                    return np.array([-1.2, 0, 0, 0, 0, 0])
                else:
                    return np.array([0, 0, 0, 1.2, 0, 0])

            # return self.preprogrammed_steps.get_wing_angles(phase)
