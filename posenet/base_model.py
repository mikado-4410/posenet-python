from abc import ABC, abstractmethod
import tensorflow as tf


class BaseModel(ABC):

    def __init__(self, sess, input_tensor_name, output_tensor_names, output_stride):
        self.output_stride = output_stride
        self.sess = sess
        self.input_tensor_name = input_tensor_name
        self.output_tensors = [
            tf.sigmoid(sess.graph.get_tensor_by_name(output_tensor_names['heatmap']), 'heatmap'),  # sigmoid!!!
            sess.graph.get_tensor_by_name(output_tensor_names['offsets']),
            sess.graph.get_tensor_by_name(output_tensor_names['displacement_fwd']),
            sess.graph.get_tensor_by_name(output_tensor_names['displacement_bwd'])
        ]

    def valid_resolution(self, width, height):
        # calculate closest smaller width and height that is divisible by the stride after subtracting 1 (for the bias?)
        target_width = (int(width) // self.output_stride) * self.output_stride + 1
        target_height = (int(height) // self.output_stride) * self.output_stride + 1
        return target_width, target_height

    @abstractmethod
    def preprocess_input(self, image):
        pass

    def predict(self, image):
        input_image, image_scale = self.preprocess_input(image)
        heatmap_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.sess.run(
            self.output_tensors,
            feed_dict={self.input_tensor_name: input_image}
        )
        return heatmap_result, offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale

    def close(self):
        self.sess.close()
