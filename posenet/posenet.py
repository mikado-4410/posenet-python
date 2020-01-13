from posenet.base_model import BaseModel


class PoseNet:

    def __init__(self, model: BaseModel):
        self.model = model

    def estimate_multiple_poses(self, image):
        heatmap_result, offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale = \
            self.model.predict(image)

        return self

    def estimate_single_pose(self, image):
        heatmap_result, offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale = \
            self.model.predict(image)

        # poses = [{'nose': {'x': 0.0, 'y': 0.0, 'score': 0}}]

        return self
