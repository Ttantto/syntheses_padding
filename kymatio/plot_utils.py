import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .backend import cdgmm, Modulus, Pad
from .utils import fft2_c2c, ifft2_c2c


def plot_wavelet_convolution(orig_im, J, L):
    im = torch.tensor(orig_im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

    M, N = im.shape[-2], im.shape[-1]
    matfilters = sio.loadmat('./matlab/filters/bumpsteerableg1_fft2d_N' + str(N) + '_J' + str(J) + '_L' + str(L) + '.mat')
    fftphi = matfilters['filt_fftphi'].astype(np.complex_)
    hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)

    fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
    hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

    hatpsi_la = torch.FloatTensor(hatpsi).cuda()  # (J,L2,M,N,2)
    hatphi = torch.FloatTensor(hatphi)  # (M,N,2)

    pad = Pad(0, False)
    x_c = pad(im)  # add zeros to imag part -> (nb,nc,M,N,2)
    hatx_c = fft2_c2c(x_c)  # fft2 -> (nb,nc,M,N,2)

    hatx_c.shape[0], hatx_c.shape[1]

    hatx_bc = hatx_c[0, 0, :, :, :]  # (M,N,2)
    hatxpsi_bc = cdgmm(hatpsi_la, hatx_bc)  # (J,L2,M,N,2)
    xpsi_bc = ifft2_c2c(hatxpsi_bc)

    xpsi_bc_mod = Modulus()(xpsi_bc).cpu().numpy()
    x, y = xpsi_bc[..., 0], xpsi_bc[..., 1]
    r = xpsi_bc.norm(p=2, dim=-1)
    theta = torch.atan2(y, x).cpu().numpy()

    wavelet_0_modulus = Modulus()(hatpsi_la).cpu().numpy()

    images = []
    for ell in range(2*L):
        l = 2*L - ell - 1
        fig = plt.figure()
        plot = fig.add_subplot(421)
        plot.imshow(orig_im)
        plot.axis('off')

        plot = fig.add_subplot(4, 2, 2)
        wavelet_0 = wavelet_0_modulus[1, l, :, :, 0]
        wavelet = np.zeros((M, N))
        wavelet[:M // 2, :N // 2] = wavelet_0[M//2:, N//2:]
        wavelet[M // 2:, N // 2:] = wavelet_0[:M // 2, :N // 2]
        wavelet[M // 2:, :N // 2] = wavelet_0[:M // 2, N // 2:]
        wavelet[:M // 2, N // 2:] = wavelet_0[M // 2:, :N // 2]
        plot.imshow(wavelet)
        plot.axis('off')

        for idx, j in enumerate((1, 3, 5)):
            plot = fig.add_subplot(4, 2, 2*idx+3)
            plot.imshow(np.abs(theta[j, l, :, :]))
            plot.axis('off')
            plot = fig.add_subplot(4, 2, 2 * idx + 4)
            plot.imshow(xpsi_bc_mod[j, l, :, :, 0])
            plot.axis('off')

        # draw the renderer
        fig.canvas.draw()
        fig.show()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)

        images.append(Image.frombytes("RGBA", (w, h), buf.tobytes()))

    images[0].save('wavelet_coeff.gif', format='GIF', append_images=images[1:], save_all=True, duration=200, loop=0)

