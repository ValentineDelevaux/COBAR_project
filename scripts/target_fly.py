import numpy as np
from .hybrid_turning_fly import HybridTurningFly

class TargetFly(HybridTurningFly):
    def __init__(self, obj_threshold=0.15, decision_interval=0.05, **kwargs):
        super().__init__(**kwargs, enable_vision=True)
        self.obj_threshold = obj_threshold
        self.decision_interval = decision_interval
        self.num_substeps = int(self.decision_interval / self.timestep)
        self.visual_inputs_hist = []

        self.coms = np.empty((self.retina.num_ommatidia_per_eye, 2))

        for i in range(self.retina.num_ommatidia_per_eye):
            mask = self.retina.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)

    def process_visual_observation(self, vision_input):
        # Calculate visual features
        features = np.zeros((2, 3))
        chasing_fly = np.zeros(2)

        for i, ommatidia_readings in enumerate(vision_input):
            is_obj = ommatidia_readings.max(axis=1) < self.obj_threshold
            is_obj_coords = self.coms[is_obj]

            if is_obj_coords.shape[0] > 0:
                features[i, :2] = is_obj_coords.mean(axis=0)
                chasing_fly[i] = is_obj.sum()

            features[i, 2] = is_obj_coords.shape[0]

        features[:, 0] /= self.retina.nrows  # normalize y_center
        features[:, 1] /= self.retina.ncols  # normalize x_center
        features[:, 2] /= self.retina.num_ommatidia_per_eye  # normalize area
        chasing_fly /= self.retina.num_ommatidia_per_eye  # normalize distance by the total number of "pixels"

        features = features.ravel().astype("float32")

        # Compare left and right visual features
        # left_deviation = 1 - features[1]
        # right_deviation = features[4]
        # left_found = features[2] > 0.005
        # right_found = features[5] > 0.005

        return features, chasing_fly