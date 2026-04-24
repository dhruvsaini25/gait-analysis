import numpy as np

class GaitFeatures:
    def __init__(self, mode="side"):
        self.mode = mode

        # SIDE VIEW
        self.left_knee = []
        self.right_knee = []
        self.left_hip = []
        self.right_hip = []

        # FRONT VIEW
        self.step_width = []
        self.hip_sway = []
        self.shoulder_sway = []

    def add_side(self, lk, rk, lh, rh):
        self.left_knee.append(lk)
        self.right_knee.append(rk)
        self.left_hip.append(lh)
        self.right_hip.append(rh)

    def add_front(self, step_w, hip_dev, shoulder_dev):
        self.step_width.append(step_w)
        self.hip_sway.append(hip_dev)
        self.shoulder_sway.append(shoulder_dev)

    def get_features(self):
        if self.mode == "side":
            return {
                "knee_asymmetry": np.mean(np.abs(np.array(self.left_knee) - np.array(self.right_knee)))
            }
        else:
            return {
                "avg_step_width": np.mean(self.step_width)
            }