from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, output_stride):
        self.output_stride = output_stride

    @abstractmethod
    def preprocess_input(self):
        pass

    @abstractmethod
    def name_output_results(self, graph):
        return graph

    def predict(self, nhwc_images):
        return nhwc_images
