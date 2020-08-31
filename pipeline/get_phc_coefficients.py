import numpy as np
import torch

from kymatio.phase_harmonics_k_bump_chunkid_simplephase import PhaseHarmonics2d
from kymatio.sufficient_stat import compute_idx_of_sufficient_stat


class ComputeCoefficentPipeline:

    def __init__(self,
                 model_name,
                 J,
                 L,
                 delta_j,
                 delta_l,
                 nb_chunks,
                 is_isotropic,
                 use_mirror_symmetry,
                 filepaths,
                 factr
                 ):
        self.J, self.L, self.delta_j, self.delta_l = J, L, delta_j, delta_l
        self.model_name = model_name
        self.filepaths = filepaths
        self.factr = factr
        self.nCov = 0
        self.count = 0
        self.Sims = []
        self.time = 0.
        self.is_isotropic = is_isotropic
        self.use_mirror_symmetry = use_mirror_symmetry
        self.nb_chunks = nb_chunks

        if self.use_mirror_symmetry and not(self.is_isotropic):
            raise Exception("You cannot use mirror symmetry if you are not isotropic")

    def run(self):
        print('Computing coefficient with model '+self.model_name+', J = '+str(self.J)+', delta_j = '+str(self.delta_j)+', delta_l = '+str(self.delta_l))
        print('Filepath '+self.filepaths[0])

        self.M, self.N = np.load(self.filepaths[0]).shape

        ###### Loading all the images:
        print("Loading the {nbr_images} images".format(nbr_images=len(self.filepaths)))
        original_ims = np.zeros((len(self.filepaths), self.M, self.N))
        for idx, filepath in enumerate(self.filepaths):
            original_im = np.load(self.filepaths[idx])
            original_im = original_im.astype('float64')
            original_ims[idx, :, :] = original_im
        original_ims_torch = torch.tensor(original_ims, dtype=torch.float, requires_grad=False)

        self.vmin = original_ims[0].mean() - 3 * original_ims[0].std()
        self.vmax = original_ims[0].mean() + 3 * original_ims[0].std()

        original_ims_torch = original_ims_torch.cuda()

        self.wph_op = PhaseHarmonics2d(
            M=self.M,
            N=self.N,
            model_name=self.model_name,
            is_isotropic=self.is_isotropic,
            J=self.J,
            L=self.L,
            delta_j=self.delta_j,
            delta_l=self.delta_l,
            delta_k=0,
            nb_chunks=self.nb_chunks,
        )

        self.wph_op.cuda()

        self.Sims = np.array([], dtype='complex')
        for chunk_id in range(self.nb_chunks + 1):
            print("Computing coeff for chunk {chunk_id} / {nb_chunks}".format(
                chunk_id=chunk_id+1, nb_chunks=self.nb_chunks+1)
            )
            Sim_ = self.wph_op(original_ims_torch, chunk_id)*self.factr  # (1,1,nb_channels,1,1,2)
            self.nCov += Sim_.shape[2]
            Sim_ = Sim_.cpu().numpy()
            Sim_ = Sim_[0, 0, :, 0, 0, 0] + 1j*Sim_[0, 0, :, 0, 0, 1]
            self.Sims = np.concatenate((self.Sims, Sim_))

        if not self.use_mirror_symmetry:
            return self.Sims

        idx_for_sufficient_stat = compute_idx_of_sufficient_stat(self.model_name, self.L, self.J, self.delta_j, self.delta_l)
        print("Starting averaging over the mirror symmetry")
        averaged_set_of_statistics= []
        for j1 in range(0, torch.max(idx_for_sufficient_stat['j1'])+1):
            indices_j1 = torch.where(idx_for_sufficient_stat['j1']==j1)[0]
            for j2 in range(0, torch.max(idx_for_sufficient_stat['j2']) + 1):
                indices_j2 = torch.where(idx_for_sufficient_stat['j2'] == j2)[0]
                for k1 in range(0, torch.max(idx_for_sufficient_stat['k1'])+1):
                    indices_k1 = torch.where(idx_for_sufficient_stat['k1'] == k1)[0]
                    for k2 in range(0, torch.max(idx_for_sufficient_stat['k2']) + 1):
                        indices_k2 = torch.where(idx_for_sufficient_stat['k2'] == k2)[0]
                        for ell in range(self.L+1):
                            indice_ell = torch.where(idx_for_sufficient_stat['ell2'] == ell)[0]
                            if len(indice_ell) == 0:
                                pass
                            if 0 < ell < self.L:
                                indice_minus_ell = torch.where(idx_for_sufficient_stat['ell2'] == 2*self.L-ell)[0]
                            else:
                                indice_minus_ell = indice_ell.new(0)
                            stat = []
                            for indice in range(len(idx_for_sufficient_stat['j1'])):
                                    if (indice in indices_j1) & (indice in indices_j2) & (indice in indices_k1) & (indice in indices_k2):
                                        if (indice in indice_ell) | (indice in indice_minus_ell):
                                            stat.append(self.Sims[indice])
                            if not ((len(stat) == 2 and ell > 0) or (ell == 0 and len(stat) == 1) or (len(stat) == 0)):
                                raise Exception("Something went wrong when averaging over the mirror symmetry")
                            if len(stat) > 0:
                                averaged_set_of_statistics.append(np.sum(stat))

        expected_length_of_averaged_set_of_statistics = len(torch.where(idx_for_sufficient_stat['ell2'] == 0)[0]) + 0.5*len(torch.where(idx_for_sufficient_stat['ell2'] > 0)[0])
        if int(expected_length_of_averaged_set_of_statistics) != len(averaged_set_of_statistics):
            raise Exception("averaged_set_of_statistics does not have the expected length")
        print("Average done: final number of coefficients: "+str(len(averaged_set_of_statistics)))

        return np.array(averaged_set_of_statistics)