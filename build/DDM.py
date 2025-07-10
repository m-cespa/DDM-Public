import numpy as np
from ImageStack import ImageStack
from typing import List
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import leastsq, curve_fit
from matplotlib.widgets import SpanSelector
import time

kB = 1.38e-23  # Boltzmann constant (J/K)
T = 297         # Temperature (K)
mu = 8.9e-4       # Viscosity (Pa.s)

class RadialAverager(object):
    """Radial average of a 2D array centred on (0,0), like the result of fft2d."""
    def __init__(self, shape):
        """A RadialAverager instance can process only arrays of a given shape, fixed at instantiation."""
        assert len(shape)==2
        # matrix of distances
        self.dists = np.sqrt(np.fft.fftfreq(shape[0])[:,None]**2 +  np.fft.fftfreq(shape[1])[None,:]**2)
        # dump the cross
        self.dists[0] = 0
        self.dists[:,0] = 0
        # discretize distances into bins
        self.bins = np.arange(max(shape)/2+1)/float(max(shape))
        # number of pixels at each distance
        self.hd = np.histogram(self.dists, self.bins)[0]
    
    def __call__(self, im):
        """Perform and return the radial average of the specrum 'im'"""
        assert im.shape == self.dists.shape
        hw = np.histogram(self.dists, self.bins, weights=im)[0]
        return hw/self.hd
    
class DDM:
    def __init__(self, filepath: str, pixel_size: float, particle_size: float, renormalise=False):
        # create the stack attribute
        self.stack = ImageStack(filepath)
        # create the numpy array preloaded stack attribute
        self.frames = self.stack.pre_load_stack(renormalise=renormalise)

        self.pixel_size = pixel_size

        # particle size only used for naming files
        self.particle_size = particle_size
        self.fps = self.stack.fps
        self.frame_count = self.stack.frame_count

    def spectrumDiff(self, im0, im1) -> np.ndarray:
        """
        Computes absolute value squared of 2D Fourier Transform of difference between im0 and im1
        Args:
            im0: 2D array for frame at time t
            im1: 2D array for frame at time t + tau
        """
        return np.abs(np.fft.fft2(im1-im0.astype(float)))**2

    def timeAveraged(self, dframes: int, maxNCouples: int=30) -> np.ndarray:
        """
        Does at most maxNCouples spectreDiff on regularly spaced couples of images. 
        Args:
            dframes: interval between frames (integer)
            maxNCouples: maximum number of couples to average over
        """
        # create array of initial times (the 'im0' in spectrumDiff) of length maxNCouples AT MOST
        # evenly spaced in increments of 'increment'
        increment = max([(self.frame_count - dframes) / maxNCouples, 1])
        initialTimes = np.arange(0, self.frame_count - dframes, increment)

        avgFFT = np.zeros(self.frames.shape[1:])
        failed = 0
        for t in initialTimes:
            if t + dframes > self.frame_count - 1:
                failed += 1
                continue

            t = np.floor(t).astype(int)

            im0 = self.frames[t]
            im1 = self.frames[t+dframes]
            if im0 is None or im1 is None:
                failed +=1
                continue
            avgFFT += self.spectrumDiff(im0, im1)
        return avgFFT / (initialTimes.size - failed)
    
    def logSpaced(self, pointsPerDecade: int=15) -> List[int]:
        """Generate an array of log spaced integers smaller than frame_count"""
        nbdecades = np.log10(self.frame_count)

        return np.unique(np.logspace(
            start=0, stop=nbdecades, 
            num=int(nbdecades * pointsPerDecade), 
            base=10, endpoint=False
            ).astype(int))
    
    def calculate_isf(self, idts: List[float], maxNCouples: int=30, plot_heat_map: bool=False) -> None:
        """
        Perform time-averaged and radial-averaged DDM for given time intervals.
        Returns ISF (Intermediate Scattering Function).

        Args:
            idts: List of integer rounded indices (within range) to specify
                which frames to time-average between
            maxNCouples: Maximum number of pairs to perform time averaging over
            n_jobs: Number of parallel jobs to run (set to -1 for all cores)
        """
        # create instance of radial averager callable
        ra = RadialAverager(self.stack.shape)

        start = time.perf_counter()
        print("\nPerforming parallelised ISF calculation...")
        
        # parallelise the time averaging
        with Parallel(n_jobs=-1, backend='threading') as parallel:
            time_avg_results = parallel(delayed(self.timeAveraged)(idt, maxNCouples) for idt in idts)
        print(f"[completed in {time.perf_counter() - start:.2f}s]")

        start = time.perf_counter()
        print("\nCalculating Radial Average for each tau time average...")

        # parallelize the radial averaging
        with Parallel(n_jobs=-1, backend='threading') as parallel:
            isf = np.array(parallel(delayed(ra)(ta) for ta in time_avg_results))
        print(f"[completed in {time.perf_counter() - start:.2f}s]")

        self.isf = isf

        qs = 2*np.pi/(2*isf.shape[-1]*self.pixel_size) * np.arange(isf.shape[-1])
        self.qs = qs

        dts = idts / self.fps
        self.dts = dts

        # if plotting feature is enabled, a heatmap will be produced
        if plot_heat_map:

            plt.figure(figsize=(5,5))
            ISF_transposed = np.transpose(isf)
            plt.imshow(ISF_transposed, cmap='viridis', aspect='auto', extent=[dts[0], dts[-1], qs[-1], qs[0]], norm=LogNorm())
            cbar = plt.colorbar(label='I(q,$\\tau$)')
            cbar.ax.yaxis.label.set_size(14)
            plt.xlabel('Lag time ($\\tau$) [s]', fontsize=16)
            plt.ylabel('Spatial Frequency (q) [$\\mu m ^{-1}$]', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{self.particle_size}μm_ISFHeatmap.png')
            plt.show()
    
    def BrownianCorrelation(self, ISF, tmax=-1, beta_guess:float=1.) -> None:
        def model_logISF(dts, A, B, tau):
            return np.log(np.maximum(A * (1 - np.exp(-dts**beta_guess / tau)) + B, 1e-10))

        nq = ISF.shape[-1]
        params = np.zeros((nq, 3))

        for iq, ddm in enumerate(ISF[:tmax].T):
            popt, _ = curve_fit(model_logISF, self.dts[:tmax], np.log(ddm), 
                                p0=[np.ptp(ISF), ddm.min(), 1], maxfev=10000)
            params[iq] = popt

        iqmin, iqmax = 0, self.qs.size - 1
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.qs, params[:, 2], 'o', label="Data")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$q\ [\mu m^{-1}]$', fontsize=16)
        ax.set_ylabel(r'Characteristic time $\tau_c\ [s]$', fontsize=16)
        ax.set_title('Select a valid q range')

        alpha_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=14, verticalalignment='top')

        def onselect(xmin, xmax):
            nonlocal iqmin, iqmax
            iqmin, iqmax = np.searchsorted(self.qs, xmin), np.searchsorted(self.qs, xmax)
            iqmax = min(iqmax, self.qs.size - 1)

            def alpha_fit(p, q, tau):
                return p[0] - p[1] * np.log(np.abs(q)) - np.log(np.abs(tau))

            alpha = leastsq(alpha_fit, [30, 2], args=(self.qs[iqmin:iqmax], params[iqmin:iqmax, 2]))[0][1]
            alpha_text.set_text(rf"$\alpha = {alpha:.2f}$")
            plt.draw()

        span = SpanSelector(ax, onselect, 'horizontal', useblit=True, interactive=True, props=dict(alpha=0.5, facecolor='red'))
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Refit after closing the interactive selection
        refined_params = np.zeros((iqmax - iqmin, 3))
        refined_covs = np.zeros((iqmax - iqmin, 3, 3))

        for iq, ddm in enumerate(ISF[:tmax].T[iqmin:iqmax]):
            popt, pcov = curve_fit(model_logISF, self.dts[:tmax], np.log(ddm), 
                                p0=[np.ptp(ISF), ddm.min(), 1], maxfev=10000)
            refined_params[iq] = popt
            refined_covs[iq] = pcov

        qs_sel, tau_sel = self.qs[iqmin:iqmax], refined_params[:, 2]
        tau_sel_err = np.sqrt(refined_covs[:, 2, 2])
        
        # scipy curve_fit will autocalculate least squares errors
        # these do NOT factor in any measurement uncertainties
        def fit_D(q, tau, tau_err):
            def model_D(q, logD):
                return logD - 2 * np.log(q)

            sigma_log_tau = tau_err / tau
            popt, pcov = curve_fit(model_D, q, np.log(tau), p0=[30], sigma=sigma_log_tau, absolute_sigma=True)

            logD, D = popt[0], np.exp(-popt[0])
            D_std_err = D * np.sqrt(pcov[0,0])
            return D, D_std_err

        D, D_std_err = fit_D(qs_sel, tau_sel, tau_sel_err)
        D *= 1e-12
        D_std_err *= 1e-12

        predicted_a = kB * T / (3 * np.pi * mu * D) * 1e6
        predicted_a_std_err = (kB * T / (3 * np.pi * mu * D**2)) * D_std_err * 1e6

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(qs_sel, tau_sel, yerr=tau_sel_err, fmt='o', label="Refined Data")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$q\ [\mu m^{-1}]$', fontsize=16)
        ax.set_ylabel(r'Characteristic time $\tau_c\ [s]$', fontsize=16)
        # ax.set_title('Final Fit with Propagated Errors')

        ax.text(0.95, 0.95, rf"$D = {D:.2e} \pm {D_std_err:.2e}\, \mathrm{{m^2/s}}$", 
                transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right')
        ax.text(0.95, 0.90, rf"a = {predicted_a:.2f} ± {predicted_a_std_err:.2f} μm", 
                transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right')

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.particle_size}_ddm_brownian.png")
        plt.show()
