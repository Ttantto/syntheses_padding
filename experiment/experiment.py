import os
from pipeline import pixelwise_phase_harmonique

k = 2 # image index

kwargs_pipeline = {
    "J": 7,
    "L": 8,
    "delta_j": 6,
    "delta_l": 4,
    "delta_n": 0,
    "nb_chunks": 20,
    "nb_restarts": 1,
    "nb_iter": 200,
    "intermediate_step": 1,
    "plot_intermediate_step": False,
    "is_isotropic": False,
    "factr": 1e7,
    "number_synthesis": 1,
    'result_path': os.path.join('result', 'patch_'+str(k), '2'),
    'filepaths': [os.path.join('data', 'raw_patches', 'patch_' + str(k) + '.npy')],
    'model_name': "model_Mopt_dn",
    "scaling_function_moments": [0, 1, 2, 3,],
    "scaling_function_file": os.path.join('bump_scaling_function', 'filters', 'scaling_niall_J_7.npy'),
}
pipeline = pixelwise_phase_harmonique.PixelWisePipeline(**kwargs_pipeline)
pipeline.run()

