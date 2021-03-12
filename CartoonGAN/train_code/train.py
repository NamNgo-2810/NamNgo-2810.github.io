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
import network 
import utils
import loss
import guided_filter
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--total_iter", default=100000, type=int)
    parser.add_argument("--adv_train_lr", default=2e-4, type=float)
    parser.add_argument("--gpu_fraction", default=0.5, type=float)
    parser.add_argument("--save_dir", default='train_cartoon', type=str)
    parser.add_argument("--use_enhance", default=False)

    args = parser.parse_args()

    return args


def train(args):
    input_photo = array_ops.placeholder(tf.float32, [args.batch_size,
                                                     args.patch_size, args.patch_size, 3])
    input_superpixel = array_ops.placeholder(tf.float32, [args.batch_size,
                                                          args.patch_size, args.patch_size, 3])
    input_cartoon = array_ops.placeholder(tf.float32, [args.batch_size,
                                                       args.patch_size, args.patch_size, 3])

    output = network.unet_generator(input_photo)
    output = guided_filter.guided_filter(input_photo, output, r=1)

    blur_fake = guided_filter.guided_filter(output, output, r=5, eps=2e-1)
    blur_cartoon = guided_filter.guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

    gray_fake, gray_cartoon = utils.color_shift(output, input_cartoon)

    d_loss_gray, g_loss_gray = loss.lsgan_loss(network.disc_sn, gray_cartoon, gray_fake,
                                               scale=1, patch=True, name='disc_gray')
    d_loss_blur, g_loss_blur = loss.lsgan_loss(network.disc_sn, blur_cartoon, blur_fake,
                                               scale=1, patch=True, name='disc_blur')

    vgg_model = loss.Vgg19('vgg19_no_fc.npy')
    vgg_photo = vgg_model.build_conv4_4(input_photo)
    vgg_output = vgg_model.build_conv4_4(output)
    vgg_superpixel = vgg_model.build_conv4_4(input_superpixel)
    h, w, c = vgg_photo.get_shape().as_list()[1:]

    photo_loss = tf.reduce_mean(losses_impl.absolute_difference(vgg_photo, vgg_output)) / (h * w * c)
    superpixel_loss = tf.reduce_mean(losses_impl.absolute_difference \
                                         (vgg_superpixel, vgg_output)) / (h * w * c)
    recon_loss = photo_loss + superpixel_loss
    tv_loss = loss.total_variation_loss(output)

    g_loss_total = 1e4 * tv_loss + 1e-1 * g_loss_blur + g_loss_gray + 2e2 * recon_loss
    d_loss_total = d_loss_blur + d_loss_gray

    all_vars = variables.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]
    disc_vars = [var for var in all_vars if 'disc' in var.name]

    tf.summary.scalar('tv_loss', tv_loss)
    tf.summary.scalar('photo_loss', photo_loss)
    tf.summary.scalar('superpixel_loss', superpixel_loss)
    tf.summary.scalar('recon_loss', recon_loss)
    tf.summary.scalar('d_loss_gray', d_loss_gray)
    tf.summary.scalar('g_loss_gray', g_loss_gray)
    tf.summary.scalar('d_loss_blur', d_loss_blur)
    tf.summary.scalar('g_loss_blur', g_loss_blur)
    tf.summary.scalar('d_loss_total', d_loss_total)
    tf.summary.scalar('g_loss_total', g_loss_total)

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
                utils.write_batch_image(photo_scenery, args.save_dir + '/images',
                                        str(total_iter) + '_scenery_photo.jpg', 4)


if __name__ == '__main__':
    args = arg_parser()
    train(args)
