import tensorflow as tf
from tensorflow.python.ops import array_ops, variables
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
import numpy as np
from white_box.train_code import guided_filter as gf


def fast_guided_filter(lr_x, lr_y, hr_x, r=1, eps=1e-8):
    lr_x_shape = tf.shape(lr_x)
    hr_x_shape = tf.shape(hr_x)

    N = gf.tf_box_filter(tf.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)

    mean_x = gf.tf_box_filter(lr_x, r) / N
    mean_y = gf.tf_box_filter(lr_y, r) / N
    cov_xy = gf.tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    var_x = gf.tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf.image.resize(A, hr_x_shape[1:3])
    mean_b = tf.image.resize(b, hr_x_shape[1:3])

    output = mean_A * hr_x + mean_b

    return output


if __name__ == '__main__':
    import cv2
    from tqdm import tqdm

    input_photo = array_ops.placeholder(tf.float32, [1, None, None, 3])
    output = gf.guided_filter(input_photo, input_photo, 5, eps=1)
    image = cv2.imread('output_figure1/cartoon2.jpg')
    image = image/127.5 - 1
    image = np.expand_dims(image, axis=0)

    config = config_pb2.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = session.Session(config=config)
    sess.run(variables.global_variables_initializer())

    out = sess.run(output, feed_dict={input_photo: image})
    out = (np.squeeze(out) + 1) * 127.5
    out = np.clip(out, 0, 255).astype((np.uint8))
    cv2.imwrite('output_figure1/cartoon2_filter.jpg', out)