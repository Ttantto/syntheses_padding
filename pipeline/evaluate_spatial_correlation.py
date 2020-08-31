import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from kymatio.phase_harmonics_k_bump_chunkid_simplephase import PhaseHarmonics2d


class EvaluateSpatialCorrelation:

    def __init__(self, filepath_per_model, result_path, J, L, M, N):
        """

        Args:
            filepath_per_model: a dict which keys are model names, and values are list of path where the image of the given model is sotre
            result_path: where to store the path
            J: the max J value
            L: the L value
        """
        self.filepath_per_model = filepath_per_model
        self.result_path = result_path
        self.nCov = 0
        self.count = 0
        self.Sims = []
        self.J, self.L, self.M, self.N = J, L, M, N

    def run(self):
        for j in range(3):
            for k in range(2):
                wph_op = PhaseHarmonics2d(
                    M=self.M,
                    N=self.N,
                    model_name="spatial_correlation",
                    J=self.J,
                    L=self.L,
                    delta_j=j,
                    delta_l=0,
                    delta_k=k,
                    nb_chunks=1,
                    shift1=0,
                    shift2=0
                )

                wph_op.cuda()
                for model, image_path in self.filepath_per_model.items():
                    correlations = []
                    for dn in range(11):
                        original_im = np.load(image_path)
                        original_im = original_im.astype('float64')
                        im = torch.tensor(original_im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()
                        shift = dn*(2**j)
                        wph_op.update_shift(shift, 0)
                        correlation_x = wph_op(im, 0)  # (1 ,1 ,P_c ,1 , 1, 2)
                        wph_op.update_shift(0, shift)
                        correlation_y = wph_op(im, 0)  # (1 ,1 ,P_c ,1 , 1, 2)

                        correlations.append(
                            (
                                np.mean(
                                    np.sqrt(
                                        correlation_x[0, 0, :, 0, 0, 0].cpu().numpy()**2
                                        + correlation_x[0, 0, :, 0, 0, 1].cpu().numpy()**2
                                    )
                                )
                                +
                                np.mean(
                                    np.sqrt(
                                        correlation_y[0, 0, :, 0, 0, 0].cpu().numpy()**2
                                        + correlation_y[0, 0, :, 0, 0, 1].cpu().numpy()**2
                                    )
                                )
                            ) / 2
                        )
                    plt.plot(range(11), correlations, label=model)
                    plt.legend()
                plt.title("J: "+str(j+1)+", k: "+str(k))
                plt.savefig(os.path.join(self.result_path, 'correlations-j'+str(j)+'-k'+str(k)+'.png'))
                plt.show()
                plt.clf()
