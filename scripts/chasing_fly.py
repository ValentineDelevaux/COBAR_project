from .hybrid_turning_fly import HybridTurningFly
import numpy as np
import flygym as flygym


class ChasingFly(HybridTurningFly):
    def __init__(self, other_fly, desired_distance, obj_threshold=0.15, decision_interval=0.05, **kwargs):
        super().__init__(**kwargs, enable_vision=True)
        self.other_fly = other_fly  # The other fly that this fly is following
        self.desired_distance = desired_distance  # The desired distance to maintain from the other fly
        self.obj_threshold = obj_threshold
        self.decision_interval = decision_interval
        self.num_substeps = int(self.decision_interval / self.timestep)
        self.visual_inputs_hist = []

        self.coms = np.empty((self.retina.num_ommatidia_per_eye, 2))

        for i in range(self.retina.num_ommatidia_per_eye):
            mask = self.retina.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)

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