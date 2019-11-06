import tensorflow as tf


class PCEN(object):
    def __init__(self):
        with tf.variable_scope('PCEN-layer'):
            self.s = tf.get_variable('s', (1,), initializer=tf.constant_initializer(0.025), dtype='float64')
            self.alpha = tf.get_variable('alpha', (1,), initializer=tf.constant_initializer(0.98), dtype='float64')
            self.delta = tf.get_variable('delta', (1,), initializer=tf.constant_initializer(2), dtype='float64')
            self.r = tf.get_variable('r', (1,), initializer=tf.constant_initializer(0.5), dtype='float64')
            self.eps = tf.constant(1E-6, dtype='float64')

    def iir(self, E, empty=True, last_state=None):
        frames = tf.split(E, E.shape[1], axis=1)
        m_frames = []
        if empty:
            last_state = None
        for frame in frames:
            if last_state is None:
                last_state = frame
                m_frames.append(frame)
                continue
            m_frame = (tf.constant(1, dtype='float64') - self.s) * last_state + self.s * frame
            last_state = m_frame
            m_frames.append(m_frame)
        M = tf.concat(m_frames, 1)
        return M

    def gen_pcen(self, inputs):
        M = self.iir(inputs)
        smooth = (self.eps + M) ** (- self.alpha)
        return (inputs * smooth + self.delta) ** self.r - self.delta ** self.r

