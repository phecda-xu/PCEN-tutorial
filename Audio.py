from dsp import *


class AudioFront(object):
    def __init__(self,
                 window=0.02,
                 stride=0.01,
                 nfft=512,
                 nfilter=40):
        self.frame_size = window
        self.frame_stride = stride
        self.nfft = nfft
        self.nfilter = nfilter

    @staticmethod
    def fbank(signal,
              sample_rate,
              window=0.02,
              stride=0.01,
              nfft=512,
              nfilter=40):
        frame_size = window
        frame_stride = stride
        emphasized_signal = preEmphasis(signal)
        frame_length, frames = enFrame(frame_size, frame_stride, sample_rate, emphasized_signal)
        frames = HMWindows(frame_length, frames)
        fft_frames = FFT(frames, nfft=nfft)
        pow_frames = powerSpectrum(fft_frames)
        filter_banks = melFilter(pow_frames, nfilter, sample_rate)
        return filter_banks

    def iirFilter(self, E, s, last_state=None, empty=True):
        if len(E.shape) != 3:
            E = E.reshape((1, E.shape[0], E.shape[1]))
        frames = [E[:, i:i+1, :] for i in range(E.shape[1])]
        m_frames = []
        if empty:
            last_state = None
        for frame in frames:
            if last_state is None:
                last_state = frame
                m_frames.append(frame)
                continue
            m_frame = (1 - s) * last_state + s * frame
            last_state = m_frame
            m_frames.append(m_frame)
        M = np.array(m_frames, ndmin=3).reshape(E.shape)
        return M

    def pcen(self,
             signal,
             sample_rate,
             alpha=0.98,
             delta=2,
             r=0.5,
             s=0.025,
             eps=1e-6):
        E = self.fbank(signal,
                       sample_rate,
                       window=self.frame_size,
                       stride=self.frame_stride,
                       nfft=self.nfft,
                       nfilter=self.nfilter)
        M = self.iirFilter(E, s)
        smooth = (eps + M) ** (-alpha)
        return (E * smooth + delta) ** r - delta ** r

