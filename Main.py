import soundfile as sf
import tensorflow as tf
from NetWork import PCEN
from Audio import AudioFront


def staticPcen(sig, sr):
    af = AudioFront()
    pcen_ = af.pcen(sig, sr)
    print(pcen_)
    return pcen_


def trainablePcen(sig, sr):
    af = AudioFront()
    E = af.fbank(sig, sr)
    if len(E.shape) != 3:
        E = E.reshape((1, E.shape[0], E.shape[1]))
    pcen = PCEN()
    input = tf.placeholder(shape=(1, 99, 40), name='input', dtype='float64')
    a = pcen.gen_pcen(input)
    b = pcen.s
    alp = pcen.alpha
    de = pcen.delta
    c = pcen.r
    feed_dict = {
        input: E
    }

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        re, s, alpha, delta, r = sess.run([a, b, alp, de, c], feed_dict=feed_dict)
    print(re, s, alpha, delta, r)


if __name__ == "__main__":
    wavfile = 'wav/0a2b400e_nohash_1.wav'
    sig, sr = sf.read(wavfile)
    static_pcen = staticPcen(sig, sr)
    trainablePcen(sig, sr)