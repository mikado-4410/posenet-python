from posenet.base_model import BaseModel


class ResNet(BaseModel):

    def __init__(self, output_stride):
        super().__init__(output_stride)

    def preprocess_input(self):
        return self

    def name_output_results(self, graph):
        return graph
