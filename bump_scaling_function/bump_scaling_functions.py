import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from bump_steerable_wavelet.bump_steerable_wavelet import compute_hat_phi_j

list_hatphi = []
for j in range(2, 9):
    print(j)
    fftphi = compute_hat_phi_j(j)
    list_hatphi.append(fftphi)
    # plt.imshow(np.real(np.fft.ifftshift(np.fft.ifft2(fftphi))))
    # plt.show()
    np.save(os.path.join('bump_scaling_function', 'filters', 'scaling_niall_J_{}.npy'.format(j)), np.array(list_hatphi))