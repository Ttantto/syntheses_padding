import os
import shutil
from timeit import default_timer as timer

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.optimize as opt
import torch
from torch.autograd import grad

from kymatio.phase_harmonics_k_bump_chunkid_simplephase import PhaseHarmonics2d


class PixelWisePipeline:

    def __init__(self,
                 model_name,
                 J,
                 L,
                 delta_j,
                 delta_l,
                 delta_n,
                 scaling_function_file,
                 scaling_function_moments,
                 is_isotropic,
                 nb_chunks,
                 nb_restarts,
                 nb_iter,
                 intermediate_step,
                 plot_intermediate_step,
                 filepaths,
                 result_path,
                 number_synthesis,
                 factr,
                 ):
        self.J, self.L, self.delta_j, self.delta_l, self.delta_n = J, L, delta_j, delta_l, delta_n
        self.nb_chunks, self.nb_restarts, self.nb_iter = nb_chunks, nb_restarts, nb_iter
        self.model_name = model_name
        self.intermediate_step, self.plot_intermediate_step = intermediate_step, plot_intermediate_step
        self.filepaths = filepaths
        self.result_path = result_path
        self.factr = factr
        self.name = ""
        self.title = ""
        self.should_record_timelapse = True
        self.nCov = 0
        self.count = 0
        self.Sims = []
        self.time = 0.
        self.is_isotropic = is_isotropic
        self.list_losses = []
        self.number_synthesis = number_synthesis
        self.scaling_function_moments = scaling_function_moments
        self.scaling_function_file = scaling_function_file


    def run(self):
        print('Starting pipeline with model '+self.model_name+', J = '+str(self.J)+', delta_j = '+str(self.delta_j)+', delta_l = '+str(self.delta_l)+', delta_n = '+str(self.delta_n))
        print('Filepath '+self.filepaths[0])

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        if not os.path.exists(os.path.join(self.result_path, 'code')):
            os.makedirs(os.path.join(self.result_path, 'code'))
            shutil.copyfile(os.path.join('experiment', 'experiment.py'), os.path.join(self.result_path, 'code', 'experiment.py'))
            shutil.copyfile(os.path.join('kymatio','sufficient_stat.py'),
                            os.path.join(self.result_path, 'code', 'sufficient_stat.py'))
            shutil.copyfile(os.path.join('kymatio','phase_harmonics_k_bump_chunkid_simplephase.py'),
                            os.path.join(self.result_path, 'code', 'phase_harmonics_k_bump_chunkid_simplephase.py'))
            shutil.copyfile(os.path.join('pipeline', 'pixelwise_phase_harmonique.py'),
                            os.path.join(self.result_path, 'code', 'pixelwise_phase_harmonique.py'))

        if not os.path.exists(os.path.join(self.result_path, 'original')):
            os.makedirs(os.path.join(self.result_path, 'original'))

        self.M, self.N = np.load(self.filepaths[0]).shape

        ###### Loading all the images:
        print("Loading the {nbr_images} images".format(nbr_images=len(self.filepaths)))
        original_ims = np.zeros((len(self.filepaths), self.M, self.N))
        for idx, filepath in enumerate(self.filepaths):
            original_im = np.load(self.filepaths[idx])
            original_im = original_im.astype('float64')
            original_ims[idx, :, :] = original_im
        original_ims_torch = torch.tensor(original_ims, dtype=torch.float)

        for idx in range(len(self.filepaths)):
            np.save(os.path.join(self.result_path, 'original', "original-"+str(idx)+".npy"), original_ims[idx])
            plt.hist(original_ims[idx].ravel(), bins=100)
            plt.savefig(os.path.join(self.result_path, 'original', "hist-original-"+str(idx)+".png"))
            plt.clf()
            plt.imshow(original_ims[idx], vmin=original_ims[idx].mean() - 3*original_ims[idx].std(), vmax=original_ims[idx].mean() + 3*original_ims[idx].std())
            plt.colorbar()
            plt.savefig(os.path.join(self.result_path, 'original', "original-"+str(idx)+".png"))
            plt.clf()

        self.vmin = original_ims[0].mean() - 3 * original_ims[0].std()
        self.vmax = original_ims[0].mean() + 3 * original_ims[0].std()

        plt.hist(original_ims.ravel(), bins=100)
        plt.savefig(os.path.join(self.result_path, 'original', "hist-original-all-img.png"))
        plt.clf()

        original_ims_torch = original_ims_torch.cuda()

        m = torch.mean(original_ims_torch)
        std = torch.std(original_ims_torch)

        self.wph_op = PhaseHarmonics2d(
            M=self.M,
            N=self.N,
            model_name=self.model_name,
            is_isotropic=self.is_isotropic,
            J=self.J,
            L=self.L,
            delta_j=self.delta_j,
            delta_l=self.delta_l,
            delta_n=self.delta_n,
            nb_chunks=self.nb_chunks,
            scaling_function_moments=self.scaling_function_moments,
            scaling_function_file=self.scaling_function_file,
        )

        self.wph_op.cuda()

        for chunk_id in range(self.nb_chunks + 1):
            print("Computing coeff for chunk {chunk_id} / {nb_chunks}".format(
                chunk_id=chunk_id+1, nb_chunks=self.nb_chunks+1)
            )
            Sim_ = self.wph_op(original_ims_torch, chunk_id)*self.factr  # (nb,nc,nb_channels,1,1,2)
            self.nCov += Sim_.shape[2]
            self.Sims.append(Sim_)


        for index_synthesis in range(self.nb_restarts):
            self.count = 0
            self.list_losses = []
            if index_synthesis == 0:
                self.should_record_timelapse = True   # we record timelapse only for first synthesis
            else:
                self.should_record_timelapse = False
            print("starting synthesis of {number_synthesis} images".format(number_synthesis=self.number_synthesis))
            x = m + std * torch.Tensor(self.number_synthesis, self.M, self.N).normal_(std=1).cuda()

            x0 = x.reshape(self.number_synthesis*self.M * self.N).cpu().numpy()
            x0 = np.asarray(x0, dtype=np.float64)

            self.name = self.model_name + "-J" + str(self.J) + "-dj" + str(self.delta_j) + "-L" + str(2 * self.L) + "-dl" + str(self.delta_l) + "-dn" + str(self.delta_n) + "-" + str(index_synthesis)
            self.title = self.model_name + " - J: " + str(self.J) + " - dj: " + str(self.delta_j) + " - L: " + str(2 * self.L) + " - dl: " + str(self.delta_l) + " - dn: " + str(self.delta_n) + " - nb_cov: " + str(self.nCov) + " - nb_iter: " + str(self.nb_iter)
            if self.is_isotropic:
                self.name = self.name + "iso"
                self.title = self.title + "iso"

            self.time = timer()
            time_starting_opt = self.time
            result = opt.minimize(self.fun_and_grad_conv, x0, method='L-BFGS-B', jac=True, tol=None,
                                  callback=self.callback_print,
                                  options={'maxiter': self.nb_iter, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 20, 'maxfun': self.nb_iter
                                           })
            final_loss, x_opt, niter, msg = result['fun'], result['x'], result['nit'], result['message']
            print('OPT ended with:', final_loss, niter, msg)
            print("Total time of opt: "+str(timer() - time_starting_opt))
            im_opt = torch.reshape(torch.tensor(x_opt, dtype=torch.float), (self.number_synthesis, self.M, self.N)).numpy()

            self.save_and_plot_result(im_opt, original_ims)

            if self.plot_intermediate_step and self.should_record_timelapse:
                self.build_time_lapse(original_ims[0])


    #---- Trying scipy L-BFGS ----#
    def obj_fun(self, im ,chunk_id):
        p = self.wph_op(im, chunk_id)*self.factr
        diff = p-self.Sims[chunk_id]
        loss = torch.mul(diff, diff).sum()/self.nCov
        if self.count == 0:
            print("space used to compute loss of chunk " + str(chunk_id) + " : " + str(torch.cuda.memory_allocated() / 1e9) + "Go")
        return loss

    def grad_obj_fun(self, x_gpu):
        loss = 0
        grad_err = torch.Tensor(self.number_synthesis, self.M, self.N).cuda()
        grad_err[:] = 0
        for chunk_id in range(self.nb_chunks+1):
            if self.count == 0:
                print("starting chunk: " + str(chunk_id))
            x_t = x_gpu.clone().requires_grad_(True)
            loss_t = self.obj_fun(x_t, chunk_id)
            grad_err_t, = grad([loss_t], [x_t], retain_graph=False)
            loss = loss + loss_t
            grad_err = grad_err + grad_err_t
        return loss, grad_err


    def fun_and_grad_conv(self, x):
        x_float = torch.reshape(torch.tensor(x, requires_grad=True, dtype=torch.float), (self.number_synthesis, self.M, self.N))
        x_gpu = x_float.cuda()
        loss, grad_err = self.grad_obj_fun(x_gpu)

        if self.count % self.intermediate_step == 0:
            print("Step {count}/{nb_iter}".format(count=self.count, nb_iter=self.nb_iter))
            print(loss)
            self.list_losses.append(loss)
            now = timer()
            delta_time = now - self.time
            self.time = now
            print("Time since last intermediate step: "+str(delta_time))
            if self.plot_intermediate_step and self.should_record_timelapse:
                if not os.path.exists(os.path.join(self.result_path, 'movie', self.name)):
                    os.makedirs(os.path.join(self.result_path, 'movie', self.name))
                ims_opt = torch.reshape(torch.tensor(x, dtype=torch.float), (self.number_synthesis, self.M, self.N)).numpy()
                np.save(os.path.join(self.result_path, 'movie', self.name, self.name + "-" + str(self.count)+'.npy'), ims_opt[0, :, :])
        self.count += 1
        return loss.cpu().item(), np.asarray(grad_err.reshape(self.number_synthesis*self.M*self.N).cpu().numpy(), dtype=np.float64)

    @staticmethod
    def callback_print(x):
        return

    def save_and_plot_result(self, ims_opt, original_ims):
        normalized_original_ims = original_ims
        normalized_ims_opt = ims_opt
        plt.clf()
        plt.plot(np.log10(self.list_losses))
        plt.savefig(os.path.join(self.result_path, "loss_"+self.name+".png"))
        plt.clf()
        for idx in range(normalized_ims_opt.shape[0]):
            im_opt = normalized_ims_opt[idx, :, :]
            plt.imshow(im_opt, vmin=im_opt.mean() - 3 * im_opt.std(), vmax=im_opt.mean() + 3 * im_opt.std())
            plt.colorbar()
            plt.title(self.title)
            plt.savefig(os.path.join(self.result_path, self.name+"-"+str(idx)+".png"))
            plt.clf()

            _, bins, _ = plt.hist(normalized_original_ims[0, :, :].ravel(), alpha=0.5, facecolor='g', label='orig', bins=100, density=True)
            plt.hist(im_opt.ravel(), alpha=0.5, facecolor='r', label='synth', bins=bins, density=True)
            plt.legend(loc='upper right')
            plt.title(self.title)
            plt.savefig(os.path.join(self.result_path, "hist-"+self.name+"-"+str(idx)+".png"))
            plt.clf()

            np.save(os.path.join(self.result_path, self.name+"-"+str(idx)+'.npy'), im_opt)

        _, bins, _ = plt.hist(normalized_original_ims.ravel(), alpha=0.5, facecolor='g', label='orig', bins=100)
        plt.hist(normalized_ims_opt.ravel(), alpha=0.5, facecolor='r', label='synth', bins=bins)
        plt.legend(loc='upper right')
        plt.title(self.title)
        plt.savefig(os.path.join(self.result_path, "hist-" + self.name + "-all_synth.png"))
        plt.clf()


    def build_time_lapse(self, original_im):
        pil_images = []
        pil_hists = []
        cv2_images = []
        cv2_hists = []

        for iter in range(0, self.nb_iter+1, self.intermediate_step):
            image_path = os.path.join(
                self.result_path,
                'movie',
                self.name,
                self.name + "-" + str(iter)+'.npy'
            )
            if not os.path.isfile(image_path):
                break
            image = np.load(image_path)

            fig = plt.figure()
            plt.imshow(image, vmin=self.vmin, vmax=self.vmax)
            plt.title(self.title)
            plt.colorbar()
            fig.canvas.draw()

            w, h = fig.canvas.get_width_height()
            buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
            plt.clf()
            plt.close(fig)
            buf.shape = (w, h, 4)

            # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
            buf = np.roll(buf, 3, axis=2)

            pil_images.append(Image.frombytes("RGBA", (w, h), buf.tobytes()))
            cv2_images.append(cv2.cvtColor(np.array(pil_images[-1]), cv2.COLOR_RGB2BGR))


            fig = plt.figure()
            values, bins, _ = plt.hist(original_im.ravel(), alpha=0.5, facecolor='g', label='orig', bins=100)
            plt.hist(image.ravel(), alpha=0.5, facecolor='r', label='synth', bins=bins)
            plt.legend(loc='upper right')
            plt.ylim(top=1.5*np.max(values))
            plt.title(self.title)
            fig.canvas.draw()

            w, h = fig.canvas.get_width_height()
            buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
            plt.clf()
            plt.close(fig)
            buf.shape = (w, h, 4)
            # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
            buf = np.roll(buf, 3, axis=2)

            pil_hists.append(Image.frombytes("RGBA", (w, h), buf.tobytes()))
            cv2_hists.append(cv2.cvtColor(np.array(pil_hists[-1]), cv2.COLOR_RGB2BGR))

        shutil.rmtree(
            os.path.join(
                self.result_path,
                'movie',
                self.name
            )
        )

        pil_images[0].save(
            os.path.join(self.result_path, 'movie', 'DG-' + self.name + '.gif'),
            format='GIF',
            append_images=pil_images[1:],
            save_all=True,
            duration=200,
            loop=0
        )
        pil_hists[0].save(
            os.path.join(self.result_path, 'movie', 'DG-hist-' + self.name + '.gif'),
            format='GIF',
            append_images=pil_hists[1:],
            save_all=True,
            duration=200,
            loop=0
        )

        height, width, layers = cv2_images[0].shape
        size = (width, height)
        out = cv2.VideoWriter(
            os.path.join(self.result_path, 'movie', 'DG-' + self.name + '.avi'),
            cv2.VideoWriter_fourcc(*'DIVX'),
            5,
            size)

        for cv2_image in cv2_images:
            out.write(cv2_image)
        out.release()

        height, width, layers = cv2_hists[0].shape
        size = (width, height)
        out = cv2.VideoWriter(
            os.path.join(self.result_path, 'movie', 'DG-hist-' + self.name + '.avi'),
            cv2.VideoWriter_fourcc(*'DIVX'),
            5,
            size)

        for cv2_hist in cv2_hists:
            out.write(cv2_hist)
        out.release()