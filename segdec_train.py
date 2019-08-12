from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import os.path
import re
import time
from datetime import datetime

import numpy as np
import pylab as plt
from tensorflow.contrib import slim

from input_data.image_processing import NetInputProcessing


class SegDecTrain(object):
    # Constants dictating the learning rate schedule.
    RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
    RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
    RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

    def __init__(self, model, storage_dir, run_string, image_size, batch_size,
                 learning_rate = 0.01,
                 max_epochs = 1000,
                 max_steps = 10000000,
                 num_gpus = 1,
                 visible_device_list = None,
                 num_preprocess_threads = 1,
                 pretrained_model_checkpoint_path = None,
                 train_segmentation_net = True,
                 train_decision_net = False,
                 use_random_rotation=False,
                 ensure_posneg_balance=True):

        self.model = model

        run_train_string = run_string[0] if type(run_string) is tuple else run_string
        run_eval_string = run_string[1] if type(run_string) is tuple else run_string

        self.visible_device_list = visible_device_list
        self.batch_size = batch_size
        self.train_dir = os.path.join(storage_dir, 'segdec_train', run_train_string) # Directory where to write event logs and checkpoint.
        self.eval_dir = os.path.join(storage_dir, 'segdec_eval', run_eval_string)

        # Takes number of learning batch iterations based on min(self.max_steps, self.max_epoch * num_batches_per_epoch)
        self.max_steps = max_steps  # Number of batches to run.
        self.max_epochs = max_epochs  # Number of epochs to run

        # Flags governing the hardware employed for running TensorFlow.
        self.num_gpus = num_gpus  # How many GPUs to use.
        self.log_device_placement = False  # Whether to log device placement

        self.num_preprocess_threads = num_preprocess_threads
        # Flags governing the type of training.
        self.fine_tune = False  # If set, randomly initialize the final layer of weights in order to train the network on a new task.
        self.pretrained_model_checkpoint_path = pretrained_model_checkpoint_path  # If specified, restore this pretrained model before beginning any training.

        self.initial_learning_rate = learning_rate  # Initial learning rate.
        self.decay_steps = 0 # no decay by default
        self.learning_rate_decay_factor = 1

        self.TOWER_NAME = "tower"

        # Batch normalization. Constant governing the exponential moving average of
        # the 'global' mean and variance for all activations.
        self.BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

        # The decay to use for the moving average.
        self.MOVING_AVERAGE_DECAY = 0.9999

        # Override the number of preprocessing threads to account for the increased
        # number of GPU towers.
        input_num_preprocess_threads = self.num_preprocess_threads * self.num_gpus

        self.input = NetInputProcessing(batch_size=self.batch_size,
                                        num_preprocess_threads=input_num_preprocess_threads,
                                        input_size=image_size,
                                        mask_size=(image_size[0],image_size[1],1),
                                        use_random_rotation=use_random_rotation,
                                        ensure_posneg_balance=ensure_posneg_balance)

        self.train_segmentation_net = train_segmentation_net
        self.train_decision_net = train_decision_net

        assert self.batch_size == 1, "Only batch_size=1 is allowed due to the way the batch_norm is used to normalize features in testing !!!"

        self.loss_print_step = 11
        self.summary_step = 110
        self.checkpoint_step = 10007

    def _tower_loss(self, images, masks, num_classes, scope, reuse_variables=None):
      """Calculate the total loss on a single tower running the ImageNet model.
    
      We perform 'batch splitting'. This means that we cut up a batch across
      multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
      then each tower will operate on an batch of 16 images.
    
      Args:
        images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                           FLAGS.image_size, 3].
        labels: 1-D integer Tensor of [batch_size].
        num_classes: number of classes
        scope: unique prefix string identifying the ImageNet tower, e.g.
          'tower_0'.
    
      Returns:
         Tensor of shape [] containing the total loss for a batch of data
      """
      # When fine-tuning a model, we do not restore the logits but instead we
      # randomly initialize the logits. The number of classes in the output of the
      # logit is the number of classes in specified Dataset.
      restore_logits = not self.fine_tune

      # Build inference Graph.
      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        net_model = self.model.get_inference(images, num_classes, for_training=True,
                                     restore_logits=restore_logits,
                                     scope=scope)

      # Build the portion of the Graph calculating the losses. Note that we will
      # assemble the total_loss using a custom function below.
      split_batch_size = images.get_shape().as_list()[0]
      self.model.get_loss(net_model, masks,
                          batch_size=split_batch_size,
                          return_segmentation_net=self.train_segmentation_net,
                          return_decision_net=self.train_decision_net)

      # Assemble all of the losses for the current tower only.
      losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)

      # Calculate the total loss for the current tower.
      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

      # Compute the moving average of all individual losses and the total loss.
      loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
      loss_averages_op = loss_averages.apply(losses + [total_loss])

      # Attach a scalar summmary to all individual losses and the total loss; do the
      # same for the averaged version of the losses.
      for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on TensorBoard.
        loss_name = re.sub('%s_[0-9]*/' % self.TOWER_NAME, '', l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(loss_name +' (raw)', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))

      with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
      return total_loss


    def _average_gradients(self, tower_grads):
      """Calculate the average gradient for each shared variable across all towers.
    
      Note that this function provides a synchronization point across all towers.
    
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
      average_grads = []
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
      return average_grads


    def train(self, dataset):
      """Train on input_data for a number of steps."""
      with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.

        # Decay the learning rate exponentially based on the number of steps.
        if self.decay_steps > 0:
            lr = tf.train.exponential_decay(self.initial_learning_rate,
                                        global_step,
                                        self.decay_steps,
                                        self.learning_rate_decay_factor,
                                        staircase=True)
        else:
            lr = self.initial_learning_rate

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        # Get images and labels for ImageNet and split the batch across GPUs.
        assert self.batch_size % self.num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')

        images, masks, _ = self.input.add_inputs_nodes(dataset, True)


        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes() + 1

         # Split the batch of images and labels for towers.
        images_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=images)
        masks_splits = tf.split(axis=0, num_or_size_splits=self.num_gpus, value=masks)

        # Calculate the gradients for each model tower.
        tower_grads = []
        reuse_variables = None
        for i in range(self.num_gpus):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % (self.TOWER_NAME, i)) as scope:
              # Force all Variables to reside on the CPU.
              with slim.arg_scope([slim.variable], device='/cpu:0'):
                # Calculate the loss for one tower of the ImageNet model. This
                # function constructs the entire ImageNet model but shares the
                # variables across all towers.
                loss = self._tower_loss(images_splits[i], masks_splits[i], num_classes,
                                   scope, reuse_variables)

              # Reuse variables for the next tower.
              reuse_variables = True

              # Retain the summaries from the final tower.
              summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

              # Retain the Batch Normalization updates operations only from the
              # final tower. Ideally, we should grab the updates from all towers
              # but these stats accumulate extremely fast so we can ignore the
              # other stats from the other towers without significant detriment.
              batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

              # Calculate the gradients for the batch of data on this ImageNet
              # tower.
              grads = opt.compute_gradients(loss)

              # Keep track of the gradients across all towers.
              tower_grads.append(grads)

        variables_to_average = (tf.trainable_variables() +
                                    tf.moving_average_variables())

        # if decision_net is not trained then remove all gradients for decision
        if self.train_decision_net is False:
            tower_grads = [[g for g in tg if g[1].name.find('decision') < 0] for tg in tower_grads]

            variables_to_average = [v for v in variables_to_average if v.name.find('decision') < 0]

        # if segmentation_net is not trained then remove all gradients for segmentation net
        # i.e. we assume all variables NOT flaged as decision net are segmentation net
        if self.train_segmentation_net is False:
            tower_grads = [[g for g in tg if g[1].name.find('decision') >= 0] for tg in tower_grads]

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = self._average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)

        # Another possibility is to use tf.slim.get_variables().
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)

        # Add summaries and visualization
        
        
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
          summaries.append(tf.summary.histogram(var.op.name, var))

        # Add weight visualization
        weight_variables = [v for v in tf.global_variables() if v.name.find('/weights') >= 0]

        for c in ['conv1_1','conv1_2',
                  'conv2_1', 'conv2_2', 'conv2_3',
                  'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4']:
            with tf.name_scope(c):
                w = [v for v in weight_variables if v.name.find('/' + c + '/') >= 0]
                w = w[0]

                x_min = tf.reduce_min(w)
                x_max = tf.reduce_max(w)
                ww = (w - x_min) / (x_max - x_min)

                ww_t = tf.transpose(ww, [3, 0, 1, 2])
                ww_t = tf.reshape(ww_t[:,:,:,0], [int(ww_t.shape[0]), int(ww_t.shape[1]), int(ww_t.shape[2]), 1])
                tf.summary.image(c, ww_t, max_outputs=10)

                summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, c))

        # Add a summaries for the input processing and global_step.
        summaries.extend(input_summaries)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
          if grad is not None:
            summaries.append(
                tf.summary.histogram(var.op.name + '/gradients', grad))

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)


        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        c = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=self.log_device_placement)
        if self.visible_device_list is not None:
            c.gpu_options.visible_device_list = self.visible_device_list
        c.gpu_options.allow_growth = True

        sess = tf.Session(config=c)
        sess.run(init)

        # restore weights from previous model
        if self.pretrained_model_checkpoint_path is not None:
            ckpt = tf.train.get_checkpoint_state(self.pretrained_model_checkpoint_path)
            if ckpt is None:
                raise Exception('No valid saved model found in ' + self.pretrained_model_checkpoint_path)

            self.model.restore(sess, ckpt.model_checkpoint_path)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(
            self.train_dir,
            graph=sess.graph)

        num_steps = min(int(self.max_epochs * dataset.num_examples_per_epoch() /  self.batch_size),
                        self.max_steps)

        prev_duration = None

        for step in range(num_steps):

          run_nodes = [train_op, loss]

          if step % self.summary_step == 0:
              run_nodes = [train_op, loss, summary_op]

          start_time = time.time()
          output_vals = sess.run(run_nodes)
          duration = time.time() - start_time

          if prev_duration is None:
              prev_duration = duration

          loss_value = output_vals[1]

          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

          if step % self.loss_print_step == 0:
            examples_per_sec = self.batch_size / float(prev_duration)
            format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, prev_duration))

          if step % self.summary_step == 0:
            summary_str = output_vals[2]
            summary_writer.add_summary(summary_str, step)

          # Save the model checkpoint periodically.
          if step % self.checkpoint_step == 0 or (step + 1) == num_steps:
            checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

          prev_duration = duration

    def _eval_once(self, eval_dir, variables_to_restore, net_op, decision_op, images_op, labels_op, img_names_op, num_examples, plot_results=True):
        """Runs Eval once.
  
        Args:
          saver: Saver.
          summary_writer: Summary writer.
          net_op: net operation with prediction          
          summary_op: Summary op.
        """
        c = tf.ConfigProto()
        if self.visible_device_list is not None:
            c.gpu_options.visible_device_list = self.visible_device_list
        c.gpu_options.allow_growth = True
        with tf.Session(config=c) as sess:
            ckpt = tf.train.get_checkpoint_state(self.train_dir)
            if ckpt and ckpt.model_checkpoint_path:

                model_checkpoint_path = ckpt.model_checkpoint_path

                # Restores from checkpoint with relative path.
                if os.path.isabs(model_checkpoint_path):
                    model_checkpoint_path = os.path.join(self.train_dir, model_checkpoint_path)

                self.model.restore(sess, model_checkpoint_path, variables_to_restore)

                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/imagenet_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Successfully loaded model from %s at step=%s.' %
                      (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                num_iter = int(math.ceil(num_examples / self.batch_size))

                # Counts the number of correct predictions.
                samples_outcome = []
                samples_names = []
                samples_speed_eval = []

                total_sample_count = num_iter * self.batch_size
                step = 0

                print('%s: starting evaluation on (%s).' % (datetime.now(), ''))
                start_time = time.time()
                while step < num_iter and not coord.should_stop():
                    start_time_run = time.time()
                    if decision_op is None:
                        predictions, image, label, img_name = sess.run([net_op, images_op, labels_op, img_names_op])
                    else:
                        predictions, decision, image, label, img_name = sess.run([net_op, decision_op, images_op, labels_op, img_names_op])

                        decision = 1.0/(1+np.exp(-np.squeeze(decision)))

                    # if we use sigmoid cross-correlation loss, then we need to add sigmoid to predictions
                    # since this is usually handled by loss which we do not use in inference
                    if self.model.use_corss_entropy_seg_net:
                        predictions = 1.0/(1+np.exp(-predictions))

                    end_time_run = time.time()

                    name = str(img_name[0]).replace("/", "_")
                    samples_names.append(name)

                    np.save(str.format("{0}/result_{2}.npy", eval_dir, step, name), predictions)
                    np.save(str.format("{0}/result_{2}_gt.npy", eval_dir, step, name), label)

                    if plot_results:
                        plt.figure(1)
                        plt.clf()
                        plt.subplot(1, 3, 1)
                        plt.title('Input image')
                        plt.imshow(image[0, :, :, 0], cmap="gray")

                        plt.subplot(1, 3, 2)
                        plt.title('Groundtruth')
                        plt.imshow(label[0, :, :, 0], cmap="gray")

                        plt.subplot(1, 3, 3)
                        if decision_op is None:
                            plt.title('Output/prediction')
                        else:
                            plt.title(str.format('Output/prediction: {0}',decision))

                        # display max
                        vmax_value = max(1, predictions.max())

                        plt.imshow((predictions[0, :, :, 0] > 0) * predictions[0, :, :, 0], cmap="jet", vmax=vmax_value)
                        plt.suptitle(str(img_name[0]))

                        plt.show(block=0)

                        out_prefix = ''

                        if decision_op is not None:
                            out_prefix = '%.3f_' % decision

                        plt.savefig(str.format("{0}/{1}result_{2}.pdf", eval_dir, out_prefix, name), bbox_inches='tight')

                    samples_speed_eval.append(end_time_run - start_time_run)

                    if decision_op is None:
                        pass
                    else:
                        samples_outcome.append((decision, np.max(label)))

                    step += 1
                    if step % 20 == 0:
                        duration = time.time() - start_time
                        sec_per_batch = duration / 20.0
                        examples_per_sec = self.batch_size / sec_per_batch
                        print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                              'sec/batch)' % (datetime.now(), step, num_iter,
                                              examples_per_sec, sec_per_batch))
                        start_time = time.time()

                if len(samples_outcome) > 0:
                    from sklearn.metrics import precision_recall_curve, roc_curve, auc

                    samples_outcome = np.matrix(np.array(samples_outcome))

                    idx = np.argsort(samples_outcome[:,0],axis=0)
                    idx = idx[::-1]
                    samples_outcome = np.squeeze(samples_outcome[idx, :])
                    samples_names = np.array(samples_names)
                    samples_names = samples_names[idx]

                    np.save(str.format("{0}/samples_outcome.npy", eval_dir), samples_outcome)
                    np.save(str.format("{0}/samples_names.npy", eval_dir), samples_names)

                    P = np.sum(samples_outcome[:, 1])

                    TP = np.cumsum(samples_outcome[:, 1] == 1).astype(np.float32).T
                    FP = np.cumsum(samples_outcome[:, 1] == 0).astype(np.float32).T

                    recall = TP / P
                    precision = TP / (TP + FP)

                    f_measure = 2 * np.multiply(recall, precision) / (recall + precision)


                    idx = np.argmax(f_measure)

                    best_f_measure = f_measure[idx]
                    best_thr = samples_outcome[idx,0]
                    best_FP = FP[idx]
                    best_FN = P - TP[idx]

                    precision_, recall_, thresholds = precision_recall_curve(samples_outcome[:, 1], samples_outcome[:, 0])
                    FPR, TPR, _ = roc_curve(samples_outcome[:, 1], samples_outcome[:, 0])
                    AUC = auc(FPR,TPR)
                    AP = auc(recall_, precision_)

                    print('AUC=%f, and AP=%f, with best thr=%f at f-measure=%.3f and FP=%d, FN=%d' % (AUC, AP, best_thr, best_f_measure, best_FP, best_FN))

                    plt.figure(1)
                    plt.clf()
                    plt.plot(recall, precision)
                    plt.title('Average Precision=%.4f' % AP)
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.savefig(str.format("{0}/precision-recall.pdf", eval_dir), bbox_inches='tight')

                    plt.figure(1)
                    plt.clf()
                    plt.plot(FPR, TPR)
                    plt.title('AUC=%.4f' % AUC)
                    plt.xlabel('False positive rate')
                    plt.ylabel('True positive rate')
                    plt.savefig(str.format("{0}/ROC.pdf", eval_dir), bbox_inches='tight')




            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

        return samples_outcome,samples_names, samples_speed_eval

    def evaluate(self, dataset, run_once = True, eval_interval_secs = 5, plot_results=True):
        """Evaluate model on Dataset for a number of steps."""
        with tf.Graph().as_default():
            # Get images and labels from the input_data.
            images, labels, img_names = self.input.add_inputs_nodes(dataset, False)

            # Number of classes in the Dataset label set plus 1.
            # Label 0 is reserved for an (unused) background class.
            num_classes = dataset.num_classes() + 1

            # Build a Graph that computes the logits predictions from the
            # inference model.
            with tf.name_scope('%s_%d' % (self.TOWER_NAME, 0)) as scope:
                net, decision,  _ = self.model.get_inference(images, num_classes, scope=scope)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(self.model.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

            eval_dir = os.path.join(self.eval_dir, dataset.subset)
            try:
                os.makedirs(eval_dir)
            except:
                pass

            while True:
                samples_outcome,samples_names, samples_speed_eval = self._eval_once(eval_dir, variables_to_restore, net, decision, images, labels, img_names, dataset.num_examples_per_epoch(),plot_results)
                if run_once:
                    break
                time.sleep(eval_interval_secs)

            num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        return samples_outcome,samples_names, samples_speed_eval,num_params


import tensorflow as tf

from segdec_model import SegDecModel
from segdec_data import InputData

if __name__ == '__main__':

    import argparse, glob, shutil

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()

    # add boolean parser to allow using 'false' in arguments
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--folds',type=str, help="Comma delimited list of ints identifying which folds to use.")
    parser.add_argument('--gpu', type=str, help="Comma delimited list of ints identifying which GPU ids to use.")
    parser.add_argument('--storage_dir', help='Path to your storage dir where segdec_train (tensorboard info) and segdec_eval (results) will be stored.',
                        type=str,
                        default='/opt/workspace/host_storage_hdd/')
    parser.add_argument('--dataset_dir', help='Path to your input_data dirs.',
                        type=str,
                        default='/opt/workspace/host_storage_hdd/')
    parser.add_argument('--datasets', help='Comma delimited list of input_data names to use, e.g., "Dataset1,Dataset2".',
                        type=str, default=','.join(['KolektorSDD']))
    parser.add_argument('--name_prefix',type=str, default=None)
    parser.add_argument('--train_subset', type=str, default="train_pos")
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--pretrained_main_folder', type=str, default=None)

    parser.add_argument('--size_height', type=int, default=2*704)
    parser.add_argument('--size_width', type=int, default=2*256)

    parser.add_argument('--seg_net_type', type=str, default='MSE')

    parser.add_argument('--input_rotation', type='bool', default=False)

    parser.add_argument('--with_seg_net', type='bool', default=True)
    parser.add_argument('--with_decision_net', type='bool', default=False)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--max_steps', type=int, default=6600)

    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--pos_weights', type=float, default=1)

    parser.add_argument('--ensure_posneg_balance', type='bool', default=True)

    args = parser.parse_args()

    main_storage_dir = args.storage_dir
    main_dataset_folder = args.dataset_dir
    dataset_list = args.datasets.split(",")
    train_subset = args.train_subset
    pretrained_model = args.pretrained_model
    pretrained_main_folder = args.pretrained_main_folder
    pos_weights = args.pos_weights
    ensure_posneg_balance = args.ensure_posneg_balance

    size_height = args.size_height
    size_width = args.size_width
    channels = args.channels

    seg_net_type = args.seg_net_type

    input_rotation = args.input_rotation

    with_seg_net = args.with_seg_net
    with_decision_net = args.with_decision_net

    max_steps = args.max_steps
    lr = args.lr

    if seg_net_type == 'MSE':
        lr_val = 0.005
        use_corss_entropy_seg_net = False
    elif seg_net_type == 'ENTROPY':
        lr_val = 0.1
        use_corss_entropy_seg_net = True
    else:
        raise Exception('Unkown SEG-NET type; allowed only: \'MSE\' or \'ENTROPY\'')


    if lr > 0:
        lr_val = lr

    folds = [int(i) for i in args.folds.split(",")]
    for i in folds:
        if i >= 0:
            fold_name = 'fold_%d' % i
        else:
            fold_name = ''

        for d in dataset_list:

            run_name = os.path.join(d, fold_name if args.name_prefix is None else os.path.join(args.name_prefix, fold_name))

            dataset_folder = os.path.join(main_dataset_folder, d)
            print("running", dataset_folder, run_name)

            if with_decision_net is False:
                # use bigger lr for sigmoid_corss_correlation loss
                net_model = SegDecModel(decision_net=SegDecModel.DECISION_NET_NONE,
                                        use_corss_entropy_seg_net=use_corss_entropy_seg_net,
                                        positive_weight=pos_weights)
            else:
                # use lr=0.005 ofr mean squated error loss
                net_model = SegDecModel(decision_net=SegDecModel.DECISION_NET_FULL,
                                        use_corss_entropy_seg_net=use_corss_entropy_seg_net,
                                        positive_weight = pos_weights)
            current_pretrained_model = pretrained_model

            if current_pretrained_model is None and pretrained_main_folder is not None:
                current_pretrained_model = os.path.join(pretrained_main_folder,fold_name)

            train = SegDecTrain(net_model,
                                storage_dir=main_storage_dir,
                                run_string=run_name,
                                image_size=(size_height,size_width,channels),  # NOTE size should be dividable by 16 !!!
                                batch_size=1,
                                learning_rate=lr_val,
                                max_steps=max_steps,
                                max_epochs=1200,
                                visible_device_list=args.gpu,
                                pretrained_model_checkpoint_path=current_pretrained_model,
                                train_segmentation_net=with_seg_net,
                                train_decision_net=with_decision_net,
                                use_random_rotation=input_rotation,
                                ensure_posneg_balance=ensure_posneg_balance)

            dataset_fold_folder = os.path.join(dataset_folder,fold_name)

            # Run training
            train.train(InputData(train_subset, dataset_fold_folder))

            if with_decision_net:
                # Run evaluation on test data
                samples_outcome_test,samples_names_test, samples_speed_eval,num_params = train.evaluate(InputData('test', dataset_fold_folder))

                np.save(os.path.join(main_storage_dir, 'segdec_eval', run_name, 'test', 'results_decision_net.npy'), samples_outcome_test)
                np.save(os.path.join(main_storage_dir, 'segdec_eval', run_name, 'test', 'results_decision_net_names.npy'), samples_names_test)
                np.save(os.path.join(main_storage_dir, 'segdec_eval', run_name, 'test', 'results_decision_net_speed_eval.npy'), samples_speed_eval)

                # Copy results from test dir of specific fold into common folder for this input_data
                src_dir = os.path.join(main_storage_dir, 'segdec_eval', run_name, 'test')
                dst_dir = os.path.join(main_storage_dir, 'segdec_eval', d if args.name_prefix is None else os.path.join(d,args.name_prefix))
                for src_file in glob.glob(os.path.join(src_dir, '*.pdf')):
                    shutil.copy(src_file, dst_dir)
