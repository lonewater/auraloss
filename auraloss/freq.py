import scipy
import torch
import numpy as np
import librosa.filters
from .utils import apply_reduction

from .perceptual import SumAndDifference, FIRFilter


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, log=True, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()
        self.log = log
        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(x_mag)
            y_mag = torch.log(y_mag)
        return self.distance(x_mag, y_mag)


class STFTLoss(torch.nn.Module):
    """STFT loss module.

    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).

    Args:
        fft_size (int, optional): FFT size in samples. Default: 1024
        hop_size (int, optional): Hop size of the FFT in samples. Default: 256
        win_length (int, optional): Length of the FFT analysis window. Default: 1024
        window (str, optional): Window to apply before FFT, options include:
           ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of scaling frequency bins. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            Default: 'mean'
        device (str, optional): Place the filterbanks on specified device. Default: None

    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'. By default.
        loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        sample_rate=None,
        scale=None,
        n_bins=None,
        scale_invariance=False,
        eps=1e-8,
        output="loss",
        reduction="mean",
        device=None,
    ):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.device = device

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(log=True, reduction=reduction)
        self.linstft = STFTMagnitudeLoss(log=False, reduction=reduction)

        # setup mel filterbank
        if self.scale == "mel":
            assert sample_rate != None  # Must set sample rate to use mel scale
            assert n_bins <= fft_size  # Must be more FFT bins than Mel bins
            fb = librosa.filters.mel(sample_rate, fft_size, n_mels=n_bins)
            self.fb = torch.tensor(fb).unsqueeze(0)
        elif self.scale == "chroma":
            assert sample_rate != None  # Must set sample rate to use chroma scale
            assert n_bins <= fft_size  # Must be more FFT bins than chroma bins
            fb = librosa.filters.chroma(sample_rate, fft_size, n_chroma=n_bins)
            self.fb = torch.tensor(fb).unsqueeze(0)

        if scale is not None and device is not None:
            self.fb = self.fb.to(self.device)  # move filterbank to device

    def stft(self, x):
        """Perform STFT.
        Args:
            x (Tensor): Input signal tensor (B, T).

        Returns:
            Tensor: x_mag, x_phs
                Magnitude and phase spectra (B, fft_size // 2 + 1, frames).
        """
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=self.eps)
        )
        x_phs = torch.angle(x_stft)
        return x_mag, x_phs

    def forward(self, x, y):
        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(x.device)
        x_mag, x_phs = self.stft(x.view(-1, x.size(-1)))
        y_mag, y_phs = self.stft(y.view(-1, y.size(-1)))

        # apply relevant transforms
        if self.scale is not None:
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag ** 2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        phs_loss = torch.nn.functional.mse_loss(x_phs, y_phs) if self.w_phs else 0.0

        # combine loss terms
        loss = (
            (self.w_sc * sc_mag_loss)
            + (self.w_log_mag * log_mag_loss)
            + (self.w_lin_mag * lin_mag_loss)
            + (self.w_phs * phs_loss)
        )

        loss = apply_reduction(loss, reduction=self.reduction)

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class MelSTFTLoss(STFTLoss):
    """Mel-scale STFT loss module."""

    def __init__(
        self,
        sample_rate,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        n_mels=128,
        **kwargs,
    ):
        super(MelSTFTLoss, self).__init__(
            fft_size,
            hop_size,
            win_length,
            window,
            w_sc,
            w_log_mag,
            w_lin_mag,
            w_phs,
            sample_rate,
            "mel",
            n_mels,
            **kwargs,
        )


class ChromaSTFTLoss(STFTLoss):
    """Chroma-scale STFT loss module."""

    def __init__(
        self,
        sample_rate,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        n_chroma=12,
        **kwargs,
    ):
        super(ChromaSTFTLoss, self).__init__(
            fft_size,
            hop_size,
            win_length,
            window,
            w_sc,
            w_log_mag,
            w_lin_mag,
            w_phs,
            sample_rate,
            "chroma",
            n_chroma,
            **kwargs,
        )


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module.

    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
    """

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        sample_rate=None,
        scale=None,
        n_bins=None,
        scale_invariance=False,
        **kwargs,
    ):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window,
                    w_sc,
                    w_log_mag,
                    w_lin_mag,
                    w_phs,
                    sample_rate,
                    scale,
                    n_bins,
                    scale_invariance,
                    **kwargs,
                )
            ]

    def forward(self, x, y):
        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []

        for f in self.stft_losses:
            if f.output == "full":  # extract just first term
                tmp_loss = f(x, y)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1])
                log_mag_loss.append(tmp_loss[2])
                lin_mag_loss.append(tmp_loss[3])
                phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(x, y)

        mrstft_loss /= len(self.stft_losses)

        if f.output == "loss":
            return mrstft_loss
        else:
            return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class RandomResolutionSTFTLoss(torch.nn.Module):
    """Random resolution STFT loss module.

    See [Steinmetz & Reiss, 2020](https://www.christiansteinmetz.com/s/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf)

    Args:
        resolutions (int): Total number of STFT resolutions.
        min_fft_size (int): Smallest FFT size.
        max_fft_size (int): Largest FFT size.
        min_hop_size (int): Smallest hop size as porportion of window size.
        min_hop_size (int): Largest hop size as porportion of window size.
        window (str): Window function type.
        randomize_rate (int): Number of forwards before STFTs are randomized.
    """

    def __init__(
        self,
        resolutions=3,
        min_fft_size=16,
        max_fft_size=32768,
        min_hop_size=0.1,
        max_hop_size=1.0,
        windows=[
            "hann_window",
            "bartlett_window",
            "blackman_window",
            "hamming_window",
            "kaiser_window",
        ],
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        sample_rate=None,
        scale=None,
        n_mels=None,
        randomize_rate=1,
        **kwargs,
    ):
        super(RandomResolutionSTFTLoss, self).__init__()
        self.resolutions = resolutions
        self.min_fft_size = min_fft_size
        self.max_fft_size = max_fft_size
        self.min_hop_size = min_hop_size
        self.max_hop_size = max_hop_size
        self.windows = windows
        self.randomize_rate = randomize_rate
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_mels = n_mels

        self.nforwards = 0
        self.randomize_losses()  # init the losses

    def randomize_losses(self):
        # clear the existing STFT losses
        self.stft_losses = torch.nn.ModuleList()
        for n in range(self.resolutions):
            frame_size = 2 ** np.random.randint(
                np.log2(self.min_fft_size), np.log2(self.max_fft_size)
            )
            hop_size = int(
                frame_size
                * (
                    self.min_hop_size
                    + (np.random.rand() * (self.max_hop_size - self.min_hop_size))
                )
            )
            window_length = int(frame_size * np.random.choice([1.0, 0.5, 0.25]))
            window = np.random.choice(self.windows)
            self.stft_losses += [
                STFTLoss(
                    frame_size,
                    hop_size,
                    window_length,
                    window,
                    self.w_sc,
                    self.w_log_mag,
                    self.w_lin_mag,
                    self.w_phs,
                    self.sample_rate,
                    self.scale,
                    self.n_mels,
                )
            ]

    def forward(self, input, target):
        if input.size(-1) <= self.max_fft_size:
            raise ValueError(
                f"Input length ({input.size(-1)}) must be larger than largest FFT size ({self.max_fft_size})."
            )
        elif target.size(-1) <= self.max_fft_size:
            raise ValueError(
                f"Target length ({target.size(-1)}) must be larger than largest FFT size ({self.max_fft_size})."
            )

        if self.nforwards % self.randomize_rate == 0:
            self.randomize_losses()

        loss = 0.0
        for f in self.stft_losses:
            loss += f(input, target)
        loss /= len(self.stft_losses)

        self.nforwards += 1

        return loss


class SumAndDifferenceSTFTLoss(torch.nn.Module):
    """Sum and difference sttereo STFT loss module.

    See [Steinmetz et al., 2020](https://arxiv.org/abs/2010.10291)

    Args:
        fft_sizes (list, optional): List of FFT sizes.
        hop_sizes (list, optional): List of hop sizes.
        win_lengths (list, optional): List of window lengths.
        window (str, optional): Window function type.
        w_sum (float, optional): Weight of the sum loss component. Default: 1.0
        w_diff (float, optional): Weight of the difference loss component. Default: 1.0
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'

    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'.
        loss, sum_loss, diff_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        w_sum=1.0,
        w_diff=1.0,
        output="loss",
    ):
        super(SumAndDifferenceSTFTLoss, self).__init__()
        self.sd = SumAndDifference()
        self.w_sum = 1.0
        self.w_diff = 1.0
        self.output = output
        self.mrstft = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths, window)

    def forward(self, input, target):
        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)

        sum_loss = self.mrstft(input_sum, target_sum)
        diff_loss = self.mrstft(input_diff, target_diff)
        loss = ((self.w_sum * sum_loss) + (self.w_diff * diff_loss)) / 2

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sum_loss, diff_loss


class PerceptuallyWeightedComplexLoss(torch.nn.Module):
    """Perceptually weighted STFT difference in the complex domain
    
    No associated publication yet...
    
    Args:
        fft_size (int, optional): FFT size in samples. Default: 1024
        hop_size (int, optional): Hop size of the FFT in samples. Default: 256
        win_length (int, optional): Length of the FFT analysis window. Default: 1024
        window (str, optional): Window to apply before FFT, options include:
           ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_p (str, optional): Perceptual weighting curve applied, options include:
           ['r468', 'aw', 'cw', None]
            Default: 'r468'
        sample_rate (int, optional): Sample rate. Default: 48000
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            Default: 'mean'

    Returns:
        loss
    """

    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_p="r468",
        sample_rate=48000,
        reduction="mean",
    ):
        super(PerceptuallyWeightedComplexLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.wp = w_p
        self.sample_rate = sample_rate
        self.reduction = reduction
        self.fftUpperRef = self.fftMax()

    def fftMax(self):
        sine = np.sin(2 * np.pi * np.arange(0,self.fft_size) * (1000/self.sample_rate))

        sine = torch.from_numpy(sine)

        S = torch.stft(sine * self.window,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        return torch.max(abs(S**2)) # defined max FFT magnitude from which to scale the qTh

    def stft(self, x):
        """Perform STFT.
        Args:
            x (Tensor): Input signal tensor (B, T).

        Returns:
            Tensor: x_mag, x_phs
                Magnitude and phase spectra (B, fft_size // 2 + 1, frames).
        """
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=self.eps)
        )
        x_phs = torch.angle(x_stft)
        return x_mag, x_phs

    def r468(self, f):
        """ Calculte ITU-R 468 weighting curve
        largely following https://github.com/cinelexi/itu-r-468-weighting/blob/master/itu_r_468_weighting/constants.py
        
        Args:
            f (array): Bin frequency values (Hz) of given Fourier transform length (fft_size // 2 + 1)

        Returns:
            Array: frequency domain weighting curve (fft_size // 2 + 1)
        """
        FACTOR_GAIN_1KHZ = 10^(18.246265068039158 / 20); # using predefined gain offset according to standard

        f1 = f
        f2 = f1**2
        f3 = f1**3
        f4 = f1**4
        f5 = f1**5
        f6 = f1**6

        h1 = ((-4.7373389813783836e-24 * f6) + (2.0438283336061252e-15 * f4) - (1.363894795463638e-07 * f2) + 1)
        h2 = ((1.3066122574128241e-19 * f5) - (2.1181508875186556e-11 * f3) + (0.0005559488023498643 * f1))

        return (0.0001246332637532143 * f1) / np.sqrt(h1**2 + h2**2) * FACTOR_GAIN_1KHZ

    def aw(self, f):
        """ Calculte a weighting curve
        
        Args:
            f (array): Bin frequency values (Hz) of given Fourier transform length (fft_size // 2 + 1)

        Returns:
            Array: frequency domain weighting curve (fft_size // 2 + 1)
        """
        f2 = f**2
        f4 = f**4

        Ra = (12194**2 * f4) / ((f2 + 20.6**2) * (f2+12194**2) * np.sqrt((f2 + 107.7**2)*(f2+737.9**2)))
        Ra1000 = (12194**2 * 1000**4) / ((1000**2 + 20.6**2) * np.sqrt((1000**2 + 107.7**2)*(1000**2+737.9**2)) * (1000**2+12194**2))
        return Ra / Ra1000

    def cw(self, f):
        """ Calculte c weighting curve
        
        Args:
            f (array): Bin frequency values (Hz) of given Fourier transform length (fft_size // 2 + 1)

        Returns:
            Array: frequency domain weighting curve (fft_size // 2 + 1)
        """
        f2 = f ** 2

        Rc = (12194**2 * f2) / ((f2 + 20.6**2) * (f2+12194**2))
        Rc1000 = (12194**2 * 1000**2) / ((1000**2 + 20.6**2) * (1000**2+12194**2))
        return Rc / Rc1000 # c is our filter mag values per bin frequency

    def forward(self, x, y):
        fbw = (self.sample_rate / 2) / (self.fft_size / 2) # bin width of fft_size at sample_rate
        fc = np.arange(fbw, fbw * (self.fft_size / 2 + 1), fbw) # centre frequencies of the bins

        # following threshold in quiet following ISO/IEC11172-3:1995
        # and more specifically Jon Boley's matlab implementation https://uk.mathworks.com/matlabcentral/fileexchange/47085-psychoacoustic-model-2
        qTh = 3.64*(fc/1000)**-0.8 - 6.5*np.exp(-0.6*(fc/1000 - 3.3)**2) + 10.^-3*(fc/1000)**4 # threshold in quiet

        ref = 10*np.log10(self.fftUpperRef) - 96 # -96 is the predefined offset
        ref = 10**(ref/10) # to linear gain

        qTh = 10**(qTh/10) * ref # convert to linear gain and scale by the calculated offset

        # get weighting curve for given bin centre frequencies
        if self.wp == "r468":
            w = self.r468(fc) 
        elif self.wp == "aw":
            w = self.aw(fc)
        elif self.wp == "cw":
            w = self.cw(fc)
        elif self.wp == None:
            w = np.ones_like(fc) # if no weighting desired then all weights are 1

        x_mag, x_phase = self.stft(x)
        y_mag, y_phase = self.stft(y)

        phase_dif = x_phase - y_phase
        phase_dif = (phase_dif + np.pi) % (2 * np.pi) - np.pi # wrap difference to [-pi, pi]

        euDif = np.sqrt(y_mag ** 2 + x_mag ** 2 - 2 * y_mag * x_mag * np.cos(phase_dif))
        euDif = (euDif * w / ((x_mag + y_mag + abs(x_mag - y_mag)) + qTh))**2 # euclidean distance gets weighted by w and normalised by magnitude. qTh works as eps

        melSumWeight = 519 / (140 * (1 + fc / 700) * np.log(10)) # derrivative of mel frequency curve can be used as compensatory weigting function for summing/averaging across bins

        rowMean = np.mean(euDif * melSumWeight) # mean over bins so to compensate for bin count
        if self.reduction == "mean":
            return np.mean(rowMean) # mean over time steps/frames
        elif self.reduction == "sum":
            return np.sum(rowMean) # sum over time steps/frames
        elif self.reduction == None:
            return rowMean
