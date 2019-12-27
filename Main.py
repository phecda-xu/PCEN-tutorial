import soundfile as sf
import tensorflow as tf
from NetWork import RPCEN, FPCEN
from python_speech_features import fbank


def test_RPCEN(sig, sr, trainable=True):
    E, _ = fbank(sig, sr, winlen=0.02, winstep=0.01, nfilt=40)
    if len(E.shape) != 3:
        E = E.reshape((1, E.shape[0], E.shape[1]))
    pcen = RPCEN(trainable=trainable)
    input_x = tf.placeholder(shape=(1, 99, 40), name='input', dtype='float32')
    a = pcen.gen_pcen(input_x)
    b = pcen.s
    alp = pcen.alpha
    de = pcen.delta
    c = pcen.r
    feed_dict = {
        input_x: E
    }

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        re, s, alpha, delta, r = sess.run([a, b, alp, de, c], feed_dict=feed_dict)
    print(re, s, alpha, delta, r)


def test_FPCEN(sig, sr):
    E, _ = fbank(sig, sr, winlen=0.02, winstep=0.01, nfilt=40)
    if len(E.shape) != 3:
        E = E.reshape((1, E.shape[0], E.shape[1]))
    pcen = FPCEN()
    input_x = tf.placeholder(shape=(1, 99, 40), name='input', dtype='float32')
    a = pcen.gen_pcen(input_x)
    alp = pcen.alpha
    de = pcen.delta
    c = pcen.r
    feed_dict = {
        input_x: E
    }

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        re, alpha, delta, r = sess.run([a, alp, de, c], feed_dict=feed_dict)
    print(re, alpha, delta, r)


if __name__ == "__main__":
    wavfile = 'wav/0a2b400e_nohash_1.wav'
    sig, sr = sf.read(wavfile)
    test_RPCEN(sig, sr, trainable=True)
    print("###############################")
    test_FPCEN(sig, sr)
