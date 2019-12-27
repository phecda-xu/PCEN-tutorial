# coding:utf-8
# Date : 2019.11.22
# author: phecda <viaxu910515@163.com>
# DEC:
#     PCEN layer with three forms
# *******************************************

import math
import tensorflow as tf


class RPCEN(object):
    def __init__(self, trainable=True):
        if trainable:
            with tf.variable_scope('pcen/trainable'):
                self.s = tf.get_variable('s', (1,), initializer=tf.constant_initializer(0.025), dtype='float32')
                self.alpha = tf.get_variable('alpha', (1,), initializer=tf.constant_initializer(0.98), dtype='float32')
                self.delta = tf.get_variable('delta', (1,), initializer=tf.constant_initializer(2), dtype='float32')
                self.r = tf.get_variable('r', (1,), initializer=tf.constant_initializer(0.5), dtype='float32')
        else:
            with tf.variable_scope('pcen/static'):
                self.s = tf.constant(0.025, dtype='float32', name='s')
                self.alpha = tf.constant(0.98, dtype='float32', name='alpha')
                self.delta = tf.constant(2, dtype='float32', name='delta')
                self.r = tf.constant(0.5, dtype='float32', name='r')
        with tf.variable_scope('pcen/init'):
            self.eps = tf.constant(1E-6, dtype='float32', name='eps')

    def iir(self, E, empty=True, last_state=None):
        with tf.variable_scope('pcen/IIR'):
            frames = tf.split(E, E.shape[1], axis=1)
            m_frames = []
            if empty:
                last_state = None
            for frame in frames:
                if last_state is None:
                    last_state = frame
                    m_frames.append(frame)
                    continue
                m_frame = (tf.constant(1, dtype='float32') - self.s) * last_state + self.s * frame
                last_state = m_frame
                m_frames.append(m_frame)
            M = tf.concat(m_frames, 1, name="filter_output")
        return M

    def gen_pcen(self, inputs):
        M = self.iir(inputs)
        with tf.variable_scope('pcen/compute'):
            smooth = (self.eps + M) ** (- self.alpha)
            sub = tf.subtract((inputs * smooth + tf.abs(self.delta)) ** self.r,
                              tf.abs(self.delta) ** self.r, name="output")
        return sub


class FPCEN():
    def __init__(self, k_smoother=2):
        self.k = k_smoother
        with tf.variable_scope('fpcen/trainable'):
            self.alpha = tf.get_variable('alpha',
                                         (1,),
                                         initializer=tf.random_normal_initializer(mean=1.0, stddev=0.1),
                                         dtype='float32')
            self.delta = tf.get_variable('delta',
                                         (1,),
                                         initializer=tf.random_normal_initializer(mean=1.0, stddev=0.1),
                                         dtype='float32')
            self.r = tf.get_variable('r',
                                     (1,),
                                     initializer=tf.random_normal_initializer(mean=1.0, stddev=0.1),
                                     dtype='float32')

            self.z = tf.get_variable('z',
                                     (self.k,),
                                     initializer=tf.random_normal_initializer(mean=math.log(1/self.k), stddev=0.1),
                                     dtype='float32')
        with tf.variable_scope('fpcen/init'):
            self.eps = tf.constant(1E-6, dtype='float32', name='eps')

    def iir(self, E, s, empty=True, last_state=None):
        with tf.variable_scope('pcen/IIR'):
            frames = tf.split(E, E.shape[1], axis=1)
            m_frames = []
            if empty:
                last_state = None
            for frame in frames:
                if last_state is None:
                    last_state = frame
                    m_frames.append(frame)
                    continue
                m_frame = (tf.constant(1, dtype='float32') - s) * last_state + s * frame
                last_state = m_frame
                m_frames.append(m_frame)
            M = tf.concat(m_frames, 1, name="filter_output")
        return M

    def smoothing(self, E):
        smoother_0 = self.iir(E, 0.015)
        smoother_1 = self.iir(E, 0.08)
        w = tf.nn.softmax(self.z)
        S = w[0] * smoother_0 + w[1] * smoother_1
        return S

    def gen_pcen(self, inputs):
        M = self.smoothing(inputs)
        with tf.variable_scope('pcen/compute'):
            smooth = (self.eps + M) ** (- self.alpha)
            sub = tf.subtract((inputs * smooth + tf.abs(self.delta)) ** self.r,
                              tf.abs(self.delta) ** self.r, name="output")
        return sub