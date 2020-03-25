"""-----------------------------------------------------------------------------

Copyright (C) 2019-2020 1QBit
Contact info: Pooya Ronagh <pooya@1qbit.com>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------------
Original Stable Baslines License: 


The MIT License

Copyright (c) 2017 OpenAI (http://openai.com)
Copyright (c) 2018-2019 Stable-Baselines Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

-----------------------------------------------------------------------------"""

import gym
import tensorflow as tf
import numpy as np
from layers import ortho_init, conv_to_fc, fc
import logging

from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy, MlpLnLstmPolicy


def conv_pseudo_1d(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride[0], stride[1], 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride[0], stride[1]]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf[0], rf[1], nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format)


def rep_cnn(unscaled_images, SPIN_N, scope, **kwargs):
    """
    CNN with spin_N x k kernels
    """

    logging.debug("rep_cnn called")


    with tf.variable_scope(scope):
        scaled_images = tf.cast(unscaled_images, tf.float32)
        activ = tf.nn.leaky_relu
        h = scaled_images
        if len(h.shape) < 4:
            h = tf.expand_dims(h, axis=-1)
        hA = activ(conv_pseudo_1d(h, f'conv_over_spins', nf=64, rf=(1, h.shape[2]), stride=(1, h.shape[2]), pad='VALID', init_scale=np.sqrt(2), **kwargs))
        hB = activ(conv_pseudo_1d(h, f'conv_over_reads', nf=64, rf=(h.shape[1], 1), stride=(h.shape[1], 1), pad='VALID', init_scale=np.sqrt(2), **kwargs))

        h = tf.concat((tf.layers.flatten(hA),tf.layers.flatten(hB)), axis=1)

        h3 = conv_to_fc(h)
        return activ(fc(h3, 'fc1', nh=64, init_scale=np.sqrt(2)))


class CnnPolicyOverReps(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CnnPolicyOverReps, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False)

        with tf.variable_scope("model", reuse=reuse):
#            activ = tf.nn.leaky_relu
            activ = tf.nn.tanh

            SPIN_N = ob_space.shape[1]
            #X, processed_x = observation_input(ob_space, nbatch)
            #print(f"{X.shape}|{processed_x.shape}")

            with tf.variable_scope("model", reuse=reuse):
                h_pi = rep_cnn(self.processed_obs, SPIN_N, scope="pi", **kwargs)  #hidden, activated CNN output
                h_vf = rep_cnn(self.processed_obs, SPIN_N, scope="vf", **kwargs)  #hidden, activated CNN output
                value_fn = tf.layers.dense(h_vf, 1, name='vf')

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(h_pi, h_vf, init_scale=0.1)

            self._value_fn = value_fn
            self._setup_init()


    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                        {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                        {self.obs_ph: obs})

        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})



class CnnLnLstmPolicyOverReps(MlpLnLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, cnn_extractor=rep_cnn, **kwargs):
        super(CnnLnLstmPolicyOverReps, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, cnn_extractor=rep_cnn)


class CnnPolicyReps(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value



#class MlpPolicy(object):
#    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
#        self.pdtype = make_pdtype(ac_space)
#        with tf.variable_scope("model", reuse=reuse):
#            X, processed_x = observation_input(ob_space, nbatch)
#            activ = tf.tanh
#            processed_x = tf.layers.flatten(processed_x)
#            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=256, init_scale=np.sqrt(2)))
#            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=256, init_scale=np.sqrt(2)))
#            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=256, init_scale=np.sqrt(2)))
#            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=256, init_scale=np.sqrt(2)))
#            vf = fc(vf_h2, 'vf', 1)[:,0]
#
#            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)
#
#
#        a0 = self.pd.sample()
#        neglogp0 = self.pd.neglogp(a0)
#        self.initial_state = None
#
#        def step(ob, *_args, **_kwargs):
#            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
#            return a, v, self.initial_state, neglogp
#
#        def value(ob, *_args, **_kwargs):
#            return sess.run(vf, {X:ob})
#
#        self.X = X
#        self.vf = vf
#        self.step = step
#        self.value = value
#
