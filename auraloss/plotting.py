import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def compare_filters(iir_b, iir_a, fir_b, fs=1):

    # compute response for IIR filter
    w_iir, h_iir = scipy.signal.freqz(iir_b, iir_a, fs=fs, worN=2048)

    # compute response for FIR filter
    w_fir, h_fir = scipy.signal.freqz(fir_b, fs=fs)

    h_iir_db = 20 * np.log10(np.abs(h_iir) + 1e-8)
    h_fir_db = 20 * np.log10(np.abs(h_fir) + 1e-8)

    plt.plot(w_iir, h_iir_db, label="IIR filter")
    plt.plot(w_fir, h_fir_db, label="FIR approx. filter")
    plt.xscale("log")
    plt.ylim([-50, 10])
    plt.xlim([10, 22.05e3])
    plt.xlabel("Freq. (Hz)")
    plt.ylabel("Mag. (dB)")
    plt.legend()
    plt.grid()
    plt.show()

def compare_freqDom_filters(fc, target_mag, fir_b, fs=1):
    w_fir, h_fir = scipy.signal.freqz(fir_b, 1, fs=fs, worN=np.shape(fc)[0])

    h_fir_db = 20 * np.log10(np.abs(h_fir) + 1e-8)
    target_mag_db = 20 * np.log10(np.abs(target_mag) + 1e-8)

    plt.plot(fc, target_mag_db, label="Target filter")
    plt.plot(w_fir, h_fir_db, label="FIR approx. filter")
    plt.xscale("log")
    plt.ylim([-50, 10])
    plt.xlim([10, 22.05e3])
    plt.xlabel("Freq. (Hz)")
    plt.ylabel("Mag. (dB)")
    plt.legend()
    plt.grid()
    plt.show()