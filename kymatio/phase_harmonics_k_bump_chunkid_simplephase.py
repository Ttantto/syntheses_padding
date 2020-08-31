import os

from kymatio.sufficient_stat import compute_idx_of_sufficient_stat, MODELS_POWER_HARMONIC_0, MODELS_POWER_HARMONIC_1, MODELS_WITH_BL, MODELS_WITH_BL_NSF, MODELS_WITH_BL_SCALING_ONLY

__all__ = ['PhaseHarmonics2d']

import torch
import numpy as np
import scipy.io as sio
from .backend import SubInitSpatialMeanC, PhaseHarmonics, PowerHarmonics0, PowerHarmonics1, DivInitStd, modulus_complex, mul
from .utils import fft2_c2c, ifft2_c2c, check_symmetry_due_to_real_field


class PhaseHarmonics2d(object):
    def __init__(self,
                 M,
                 N,
                 model_name,
                 is_isotropic,
                 J,
                 L,
                 delta_j,
                 delta_l,
                 delta_n,
                 nb_chunks,
                 scaling_function_moments=[0, 1, 2, 3],
                 scaling_function_file=None,
                 devid=0,
                 ):
        self.model_name = model_name
        self.M, self.N, self.J, self.L = M, N, J, L  # size of image, max scale, number of angles [0,pi]
        self.dj = delta_j  # max scale interactions
        self.dl = delta_l  # max angular interactions
        self.dn = delta_n
        self.nb_chunks = nb_chunks  # number of chunks to cut whp cov
        self.devid = devid  # gpu id
        if self.dl > self.L:
            raise (ValueError('delta_l must be <= L'))
        self.is_isotropic = is_isotropic
        self.should_check_real_symmetry = False
        self.scaling_function_file = scaling_function_file
        self.scaling_function_moments = scaling_function_moments
        self.build()


    def build(self):
        if self.model_name in MODELS_POWER_HARMONIC_0:
            self.phase_harmonics = PowerHarmonics0.apply
        elif self.model_name in MODELS_POWER_HARMONIC_1:
            self.phase_harmonics = PowerHarmonics1.apply
        else:
            self.phase_harmonics = PhaseHarmonics.apply
        self.power_harmonic = PowerHarmonics0.apply

        self.filters_tensor()
        self.idx_wph = compute_idx_of_sufficient_stat(self.model_name, self.L, self.J, self.dj, self.dl, self.dn)
        self.wph_by_chunk = self.get_this_chunk(self.nb_chunks)
        self.subinitmean1 = {}
        self.subinitmean2 = {}
        self.divinitstd1 = {}
        self.divinitstd2 = {}

        for chunk_id in range(self.nb_chunks+1):
            if chunk_id < self.nb_chunks:
                self.subinitmean1[chunk_id] = SubInitSpatialMeanC(self.is_isotropic)
                self.subinitmean2[chunk_id] = SubInitSpatialMeanC(self.is_isotropic)
                self.divinitstd1[chunk_id] = DivInitStd(self.is_isotropic)
                self.divinitstd2[chunk_id] = DivInitStd(self.is_isotropic)
            else:
                self.subinitmeanJ = SubInitSpatialMeanC()
                self.subinitmeanJabs = SubInitSpatialMeanC()
                self.divinitstdmeanJ = DivInitStd()
                self.subinitmeanPixel = SubInitSpatialMeanC()



    def filters_tensor(self):
        assert(self.M == self.N)
        if self.model_name in MODELS_WITH_BL:
            matfilters = sio.loadmat('./BL_wavelet/filters/BL_N' + str(self.N) + '_J' + str(self.J) + '.mat')
        elif self.model_name in MODELS_WITH_BL_NSF:
            matfilters = sio.loadmat('./BL_wavelet/filters/BL_NSF_N' + str(self.N) + '_J' + str(self.J) + '.mat')
        elif self.model_name in MODELS_WITH_BL_SCALING_ONLY:
            matfilters = sio.loadmat('./BL_wavelet/filters/BL_SCALING_ONLY_N' + str(self.N) + '_J' + str(self.J) + '.mat')
        else:
            matfilters = np.load(os.path.join('bump_steerable_wavelet', 'filters', 'bump_steerable_wavelet_N_'+str(self.N)+'_J_'+str(self.J)+'_L'+str(self.L)+'_dn'+str(self.dn)+'.npy'), allow_pickle=True).item()


        fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
        hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

        if self.scaling_function_file is not None:
            fftphi = np.load(self.scaling_function_file).astype(np.complex_)
        else:
            fftphi = matfilters['filt_fftphi'].astype(np.complex_)
        hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)

        self.hatpsi = torch.FloatTensor(hatpsi)  # (L2, J, 2*delta_n+1, M,N,2)
        self.hatphi = torch.FloatTensor(hatphi)  # (M,N,2) or (J,M,N,2)


    def get_this_chunk(self, nb_chunks):
        # cut self.idx_wph into smaller pieces
        nb_cov = len(self.idx_wph['j1'])
        print("Total number of cov: "+str(nb_cov)+" times L2="+str(self.L*2))
        min_chunk = nb_cov // nb_chunks
        print("Number of cov per chunk: "+str(min_chunk)+" or "+str(min_chunk+1))
        nb_cov_chunk = np.zeros(nb_chunks, dtype=np.int32)
        for idxc in range(nb_chunks):
            nb_cov_chunk[idxc] = int(min_chunk)
        for idxc in range(nb_cov - min_chunk*nb_chunks):
            nb_cov_chunk[idxc] = nb_cov_chunk[idxc] + 1

        wph_by_chunk = dict()
        offset = int(0)
        for idxc in range(nb_chunks):
            wph_by_chunk[idxc] = {}
            wph_by_chunk[idxc]['j1'] = self.idx_wph['j1'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['j2'] = self.idx_wph['j2'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['ell2'] = self.idx_wph['ell2'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['k1'] = self.idx_wph['k1'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['k2'] = self.idx_wph['k2'][offset:offset+nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['dn1'] = self.idx_wph['dn1'][offset:offset + nb_cov_chunk[idxc]]
            wph_by_chunk[idxc]['dn2'] = self.idx_wph['dn2'][offset:offset + nb_cov_chunk[idxc]]
            offset = offset + nb_cov_chunk[idxc]

        return wph_by_chunk

    def _type(self, _type, devid=None):
        self.hatpsi = self.hatpsi.type(_type)
        self.hatphi = self.hatphi.type(_type)
        if devid is not None:
            self.hatpsi = self.hatpsi.to(devid)
            self.hatphi = self.hatphi.to(devid)
        return self

    def cuda(self):
        """
            Moves tensors to the GPU
        """
        devid = self.devid
        print('call cuda with devid=', devid)
        assert(devid>=0)
        for chunk_id in range(self.nb_chunks):
            self.wph_by_chunk[chunk_id]['j1'] = self.wph_by_chunk[chunk_id]['j1'].type(torch.cuda.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['j2'] = self.wph_by_chunk[chunk_id]['j2'].type(torch.cuda.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['dn1'] = self.wph_by_chunk[chunk_id]['dn1'].type(torch.cuda.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['dn2'] = self.wph_by_chunk[chunk_id]['dn2'].type(torch.cuda.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['ell2'] = self.wph_by_chunk[chunk_id]['ell2'].type(torch.cuda.LongTensor).to(devid)
            self.wph_by_chunk[chunk_id]['k1'] = self.wph_by_chunk[chunk_id]['k1'].type(torch.cuda.FloatTensor).to(devid)
            self.wph_by_chunk[chunk_id]['k2'] = self.wph_by_chunk[chunk_id]['k2'].type(torch.cuda.FloatTensor).to(devid)

        return self._type(torch.cuda.FloatTensor, devid)

    def cpu(self):
        """
            Moves tensors to the CPU
        """
        print('call cpu')
        return self._type(torch.FloatTensor)

    def add_padding(self, tensor_to_pad):
        padding = 2**self.J
        tensor_to_pad[:, :, :, :padding, :, :] = 0.
        tensor_to_pad[:, :, :, -padding:, :, :] = 0.
        tensor_to_pad[:, :, :, :, :padding, :] = 0.
        tensor_to_pad[:, :, :, :, -padding:, :] = 0.
        return tensor_to_pad

    def forward(self, input, chunk_id):
        M = self.M
        N = self.N
        L2 = self.L*2
        Nimg = input.size(0)
        if self.model_name in MODELS_WITH_BL:
            L2 = 3
        elif self.model_name in MODELS_WITH_BL_NSF:
            L2 = 4
        elif self.model_name in MODELS_WITH_BL_SCALING_ONLY:
            L2 = 1

        # denote
        # input: (Nimg,M,N) if the field is real or (Nimg, M, N, 2) if the field is complex
        if input.dim() == 3:
            x_c = input.new_zeros(input.size(0), input.size(1), input.size(2), 2)
            x_c[:, :, :, 0] = input  # (Nimg, M, N, 2)
        else:
            raise ValueError("The dim of the input should be 3: (Nimg, M, N)")


        if chunk_id < self.nb_chunks:
            hatx_c = fft2_c2c(x_c)  # fft2 -> (Nimg, M, N, 2)
            del x_c
            hatpsi_la_chunk, list_places_1, list_places_2 = self.create_hatpsi_la_chunk(chunk_id)
            hatx_bc = hatx_c.unsqueeze(1).unsqueeze(1) # (Nimg, 1, 1, M, N, 2)
            del hatx_c
            hatxpsi_bc = mul(hatpsi_la_chunk, hatx_bc)  # (Nimg, L2, N_in_chunk, M, N, 2)
            del hatpsi_la_chunk, hatx_bc
            xpsi_bc = ifft2_c2c(hatxpsi_bc)  # (Nimg, L2, N_in_chunk, M, N, 2)
            del hatxpsi_bc

            if self.should_check_real_symmetry:
                check_symmetry_due_to_real_field(xpsi_bc)

            # select la1, et la2, P_c = number of |la1| in this chunk
            nb_channels = self.wph_by_chunk[chunk_id]['j1'].shape[0]
            xpsi_bc_la1 = xpsi_bc.new(Nimg, L2, nb_channels, M, N, 2)  # (Nimg, L2, P_c, M, N, 2)
            xpsi_bc_la2 = xpsi_bc.new(Nimg, L2, nb_channels, M, N, 2)  # (Nimg, L2, P_c, M, N, 2)
            for ell1 in range(L2):
                ell2 = torch.remainder(self.wph_by_chunk[chunk_id]['ell2'] + ell1, L2)
                xpsi_bc_la1[:, ell1, ...] = xpsi_bc[:, ell1, list_places_1, :, :, :]
                xpsi_bc_la2[:, ell1, ...] = xpsi_bc[:, ell2, list_places_2, :, :, :]

            del xpsi_bc
            xpsi_bc_la1k1 = self.phase_harmonics(xpsi_bc_la1, self.wph_by_chunk[chunk_id]['k1'])  # (Nimg, L2, P_c, M, N, 2)
            xpsi_bc_la2k2 = self.phase_harmonics(xpsi_bc_la2, self.wph_by_chunk[chunk_id]['k2'])  # (Nimg, L2, P_c, M, N, 2)

            del xpsi_bc_la1, xpsi_bc_la2

            # Taking the complex conjugate of xpsi_bc_la2k2
            xpsi_bc_la2k2[..., 1] = - xpsi_bc_la2k2[..., 1]

            # Padding the result
            xpsi_bc_la1k1 = self.add_padding(xpsi_bc_la1k1)
            xpsi_bc_la2k2 = self.add_padding(xpsi_bc_la2k2)

            # substract spatial mean along M and N
            xpsi0_bc_la1k1 = self.subinitmean1[chunk_id](xpsi_bc_la1k1)  # (Nimg, L2, P_c, M, N, 2)
            xpsi0_bc_la2k2 = self.subinitmean2[chunk_id](xpsi_bc_la2k2)  # (Nimg, L2, P_c, M, N, 2)
            # del xpsi_bc_la1k1, xpsi_bc_la2k2

            xpsi0_bc_la1k1 = self.divinitstd1[chunk_id](xpsi0_bc_la1k1)  # (Nimg, L2, P_c, M, N, 2)
            xpsi0_bc_la2k2 = self.divinitstd2[chunk_id](xpsi0_bc_la2k2)  # (Nimg, L2, P_c, M, N, 2)


            # compute mean spatial
            corr_xpsi_bc = mul(xpsi0_bc_la1k1, xpsi0_bc_la2k2)    # (Nimg, L2, P_c, M, N, 2)
            del xpsi0_bc_la1k1, xpsi0_bc_la2k2
            corr_bc = torch.mean(torch.mean(torch.mean(corr_xpsi_bc, -2, True), -3, True), 0, True)    # (1, L2, P_c, 1, 1, 2)
            del corr_xpsi_bc

            if self.is_isotropic:
                corr_bc = torch.mean(corr_bc, 1, True)  # (1, 1, P_c, 1, 1, 2)
            else:
                corr_bc = corr_bc.view(1, 1, nb_channels * L2, 1, 1, 2)
            return corr_bc

        else:
            # ADD 2 channel for spatial phiJ evaluated against x and abs(x)
            # add l2 phiJ to last channel
            xc_0 = self.subinitmeanPixel(x_c)
            del x_c
            hatx_c = fft2_c2c(xc_0).unsqueeze(1).unsqueeze(1)  # fft2 -> (Nimg, 1, 1, M, N, 2)
            if self.hatphi.dim() == 3:
                hatphi = self.hatphi.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, M, N, 2)
            else:
                hatphi = self.hatphi.unsqueeze(1).unsqueeze(0)    # (1, J, 1, M, N, 2)

            hatxphi_c = mul(hatx_c, hatphi)  # (Nimg, J, 1, M, N, 2)
            xpsi_c = ifft2_c2c(hatxphi_c)  # (Nimg, J, 1,  M, N, 2)
            tensor_k = xpsi_c.new_tensor(self.scaling_function_moments)
            xpsi_c = xpsi_c.expand((xpsi_c.size(0), xpsi_c.size(1), len(self.scaling_function_moments), M, N, 2))
            xpsi_c_k = self.power_harmonic(xpsi_c, tensor_k)

            xpsi_c_k = self.add_padding(xpsi_c_k)

            # submean from spatial M N
            xpsi0_c = self.subinitmeanJ(xpsi_c_k)   # (Nimg, J, K, M, N, 2)
            xpsi0_c = self.divinitstdmeanJ(xpsi0_c)  # (Nimg, J, K, M, N, 2)
            xpsi0_mod = modulus_complex(xpsi0_c)  # (Nimg, J, K, M, N, 2)
            xpsi0_mod2 = mul(xpsi0_mod, xpsi0_mod)  # (Nimg, J, K, M, N, 2)
            Sout = input.new(1, 1, len(self.scaling_function_moments)*xpsi0_mod2.size(1), 1, 1, 2)
            xpsi0_mod2 = xpsi0_mod2.view(xpsi0_mod2.size(0), 1, xpsi0_mod2.size(1)*xpsi0_mod2.size(2), M, N, 2)
            Sout[:, :, :, :, :, :] = torch.mean(torch.mean(torch.mean(xpsi0_mod2, -2, True), -3, True), 0, True)
            return Sout


    def create_hatpsi_la_chunk(self, chunk_id):
        list_indices, list_places = torch.unique(
            torch.cat(
                (torch.stack((self.wph_by_chunk[chunk_id]['j1'], self.wph_by_chunk[chunk_id]['dn1']), dim=0),
                 torch.stack((self.wph_by_chunk[chunk_id]['j2'], self.wph_by_chunk[chunk_id]['dn2']), dim=0)),
                dim=1),
            dim=1, return_inverse=True
        )
        list_places_1 = list_places[:list_places.shape[0] // 2]
        list_places_2 = list_places[list_places.shape[0] // 2: ]
        hatpsi_la_chunk = self.hatpsi[:, list_indices[0], list_indices[1], :, :, :]
        return hatpsi_la_chunk.unsqueeze(0), list_places_1, list_places_2



    def __call__(self, input, chunk_id):
        return self.forward(input, chunk_id)
