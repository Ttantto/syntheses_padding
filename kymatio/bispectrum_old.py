from itertools import product

import numpy as np
import numpy.random as ra

import matplotlib.pyplot as plt
import torch

from kymatio.backend import conjugate
from .utils import fft2_c2c

def compute_total_bispectrum(data):
    bispec_shape = (int(data.size()[2] / 2.), int(data.size()[3] / 2.))
    iterator_frequencies = product(range(bispec_shape[0]), range(bispec_shape[1]))
    bispectrum = compute_bispectrum(data, iterator_frequencies)
    bispectrum.view(1, 1, bispec_shape[0], bispec_shape[1], 2)


def compute_bispectrum(
        data,
        iterator_frequencies,
        nsamples=100,
):
    '''
    Do the computation.

    Parameters
    ----------
    '''
    data_c = data.new_zeros(data.size(0), data.size(1), data.size(2), data.size(3), 2)
    data_c.select(4, 0)[:] = data  # (1,Nimg,M,N,2)
    size_bispectrum = len(iterator_frequencies)

    fftarr = fft2_c2c(data_c)
    conjfft = conjugate(fftarr)

    bispectrum = torch.Tensor(1, 1, size_bispectrum, 2)

    for n, (k1mag, k2mag) in enumerate(iterator_frequencies):
        if n % 1000 == 0:
            print(str(n) + " / " + size_bispectrum)
        phi1 = ra.uniform(0, 2 * np.pi, nsamples)
        phi2 = phi1 + (np.pi/4.)
        # phi2 = ra.uniform(0, 2 * np.pi, nsamples)

        k1x = np.asarray([int(k1mag * np.cos(angle))
                          for angle in phi1])
        k2x = np.asarray([int(k2mag * np.cos(angle))
                          for angle in phi2])
        k1y = np.asarray([int(k1mag * np.sin(angle))
                          for angle in phi1])
        k2y = np.asarray([int(k2mag * np.sin(angle))
                          for angle in phi2])

        k3x = np.asarray([int(k1mag * np.cos(ang1) +
                              k2mag * np.cos(ang2))
                          for ang1, ang2 in zip(phi1, phi2)])
        k3y = np.asarray([int(k1mag * np.sin(ang1) +
                              k2mag * np.sin(ang2))
                          for ang1, ang2 in zip(phi1, phi2)])

        samps = fftarr[:, :, k1x, k1y, ...] * fftarr[:, :, k2x, k2y, ...] * conjfft[:, :, k3x, k3y, ...]

        bispectrum[0, 0, n, :] = torch.sum(samps, dim=2)[0, 0, :]

    return bispectrum


def save_bispectrum(bispectrum, filename):
    plt.clf()
    plt.imshow(np.log(bispectrum), origin="lower", interpolation="nearest")
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(filename + "png")
    np.save(filename + 'npy', bispectrum)