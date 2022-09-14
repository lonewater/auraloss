import torch
import numpy as np
import scipy.signal


class SumAndDifference(torch.nn.Module):
    """Sum and difference signal extraction module."""

    def __init__(self):
        """Initialize sum and difference extraction module."""
        super(SumAndDifference, self).__init__()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, #channels, #samples).
        Returns:
            Tensor: Sum signal.
            Tensor: Difference signal.
        """
        if not (x.size(1) == 2):  # inputs must be stereo
            raise ValueError(f"Input must be stereo: {x.size(1)} channel(s).")

        sum_sig = self.sum(x).unsqueeze(1)
        diff_sig = self.diff(x).unsqueeze(1)

        return sum_sig, diff_sig

    @staticmethod
    def sum(x):
        return x[:, 0, :] + x[:, 1, :]

    @staticmethod
    def diff(x):
        return x[:, 0, :] - x[:, 1, :]


class FIRFilter(torch.nn.Module):
    """FIR pre-emphasis filtering module.

    Args:
        filter_type (str): Shape of the desired FIR filter ("hp", "fd", "aw"). Default: "hp"
        coef (float): Coefficient value for the filter tap (only applicable for "hp" and "fd"). Default: 0.85
        ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
        plot (bool): Plot the magnitude respond of the filter. Default: False
        inverse (bool): Return the reciprocal of the filter response for the sake of de-emphasis

    Based upon the perceptual loss pre-empahsis filters proposed by
    [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    A-weighting filter - "aw"
    C-weighting filter - "cw"
    ITU-R 468 weighting filter = "r468w"
    First-order highpass - "hp"
    Folded differentiator - "fd"

    Note that the default coefficeint value of 0.85 is optimized for
    a sampling rate of 44.1 kHz, considering adjusting this value at differnt sampling rates.
    """

    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101, plot=False, inverse=False):
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot
        self.inverse = inverse

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
            )
            DENs = np.polymul(
                np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
            )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            # use the reciprocal of the magnitude response if inverse filter required
            if inverse == True:
                taps = scipy.signal.firls(ntaps, w_iir, 1 / abs(h_iir), fs=fs)
            else:
                taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

            if plot:
                from .plotting import compare_filters
                compare_filters(b, a, taps, fs=fs)
        elif filter_type == "cw":
            fft_size = 2**12 # define a big fft for ideal freq domain representation
            fbw = (fs / 2) / (fft_size / 2) # bin width of fft_size at fs
            fc = np.arange(0, fbw * (fft_size / 2), fbw) # centre frequencies of the bins

            f2 = fc ** 2

            Rc = (12194**2 * f2) / ((f2 + 20.6**2) * (f2+12194**2))
            Rc1000 = (12194**2 * 1000**2) / ((1000**2 + 20.6**2) * (1000**2+12194**2))
            c = Rc / Rc1000 # c is our filter mag values per bin frequency
            
            # then we fit to 101 tap FIR filter with least squares
            # use the reciprocal of the magnitude response if inverse filter required
            if inverse == True:
                taps = scipy.signal.firls(ntaps, fc, 1/c, fs=fs)
            else:
                taps = scipy.signal.firls(ntaps, fc, c, fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

            if plot:
                 from .plotting import compare_freqDom_filters
                 compare_freqDom_filters(fc, c, taps, fs=fs)

        elif filter_type == "r468w":
            fft_size = 2**12 # define a big fft for ideal freq domain representation
            fbw = (fs / 2) / (fft_size / 2) # bin width of fft_size at fs
            fc = np.arange(0, fbw * (fft_size / 2), fbw) # centre frequencies of the bins

            db_gain_1kHz = 18.246265068039158 # predefined gain offset (dB)
            factor_gain_1kHz = 10**(db_gain_1kHz / 20) # to lin gain

            f1 = fc
            f2 = f1**2
            f3 = f1**3
            f4 = f1**4
            f5 = f1**5
            f6 = f1**6

            h1 = ((-4.7373389813783836e-24 * f6) + (2.0438283336061252e-15 * f4) - (1.363894795463638e-07 * f2) + 1)
            h2 = ((1.3066122574128241e-19 * f5) - (2.1181508875186556e-11 * f3) + (0.0005559488023498643 * f1))

            r468 = (0.0001246332637532143 * f1) / np.sqrt(h1**2 + h2**2) * factor_gain_1kHz # r468 is our filter mag values per bin frequency

            # then we fit to 101 tap FIR filter with least squares
            # use the reciprocal of the magnitude response if inverse filter required
            if inverse == True:
                r468 = 1/r468
                r468[0] = 0 # avoids 1/0 issue
                taps = scipy.signal.firls(ntaps, fc, r468, fs=fs)
            else:
                taps = scipy.signal.firls(ntaps, fc, r468, fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

            if plot:
                from .plotting import compare_freqDom_filters
                compare_freqDom_filters(fc, r468, taps, fs=fs)  

    def forward(self, input, target=None):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor)(optional): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """

        if target is not None:
            input = torch.nn.functional.conv1d(
                input, self.fir.weight.data, padding=self.ntaps // 2
            )

            target = torch.nn.functional.conv1d(
                target, self.fir.weight.data, padding=self.ntaps // 2
            )
            return input, target
        else:
            input = torch.nn.functional.conv1d(
                input, self.fir.weight.data, padding=self.ntaps // 2, groups=input.size()[1]
            )
            return input,