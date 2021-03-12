import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import ops
from tensorflow.python.client import session
from tensorflow.python.training import adam, training
from tensorflow.python.ops import array_ops, variables
from tensorflow.python.ops.losses import losses_impl
import os
import numpy as np
import argparse
import network, utils
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--total_iter", default=50000, type=int)
    parser.add_argument("--adv_train_lr", default=2e-4, type=float)
    parser.add_argument("--gpu_fraction", default=0.5, type=float)
    parser.add_argument("--save_dir", default='pretrain')

    return parser.parse_args()


def train(args):
    input_photo = array_ops.placeholder(tf.float32, [args.batch_size, args.patch_size, args.patch_size, 3])
    output = network.unet_generator(input_photo)
    recon_loss = tf.reduce_mean(losses_impl.absolute_difference(input_photo, output))
    all_vars = variables.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]

    update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optim = adam.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99).minimize(recon_loss, var_list=gene_vars)

    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = session.Session(config=config_pb2.ConfigProto(gpu_options=gpu_options))
    saver = training.Saver(var_list=gene_vars, max_to_keep=20)

    with tf.device('/device:GPU:0'):
        sess.run(variables.global_variables_initializer())
        face_photo_dir = 'dataset/photo_face'
        face_photo_list = utils.load_image_list(face_photo_dir)
        scenery_photo_dir = 'dataset/photo_scenery'
        scenery_photo_list = utils.load_image_list(scenery_photo_dir)

        for total_iter in tqdm(range(args.total_iter)):
            photo_batch = utils.next_batch(face_photo_list, args.batch_size) if np.mod(total_iter, 5) == 0 \
                else utils.next_batch(scenery_photo_list, args.batch_size)
            _, r_loss = sess.run([optim, recon_loss], feed_dict={input_photo: photo_batch})
            if np.mod(total_iter + 1, 50) == 0:
                saver.save(sess, args.save_dir + 'save_models/model', write_meta_graph=False, global_step=total_iter)
                photo_face = utils.next_batch(face_photo_list, args.batch_size)
                photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)

                result_face = sess.run(output, feed_dict={input_photo: photo_face})
                result_scenery = sess.run(output, feed_dict={input_photo: photo_scenery})

                utils.write_batch_image(result_face, args.save_dir + '/images', str(total_iter) + 'face_result.jpg', 4)
                utils.write_batch_image(photo_face, args.save_dir + '/images', str(total_iter) + '_face_photo.jpg', 4)
                utils.write_batch_image(result_scenery, args.save_dir + '/images', str(total_iter) + '_scenery_result'
                                                                                                     '.jpg', 4)
                utils.write_batch_image(photo_scenery, args.save_dir + '/images', str(total_iter) + '_scenery_photo.jpg', 4)


if __name__ == '__main__':
    args = arg_parser()
    train(args)