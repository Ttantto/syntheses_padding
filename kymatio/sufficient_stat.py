import torch

from .utils import periodic_dis


MODELS_POWER_HARMONIC_0 = ["model_PO"]
MODELS_POWER_HARMONIC_1 = ["model_P1", "model_BL", "model_BL_NSF", "model_BL_NSF_GAUSSIAN_PURE", "model_BL_NSF_GAUSSIAN", "model_BL_ONLY_SCALING"]
MODELS_WITH_BL = ["model_BL"]
MODELS_WITH_BL_NSF = ["model_BL_NSF", "model_BL_NSF_GAUSSIAN", "model_BL_NSF_GAUSSIAN_PURE"]
MODELS_WITH_BL_SCALING_ONLY = ["model_BL_ONLY_SCALING"]


def compute_idx_of_sufficient_stat(model_name, L, J, dj, dl, dn=0):
    if model_name == "model_A":
        return compute_idx_model_A(L, J, dn, dj, dl)
    if model_name == "model_B":
        return compute_idx_model_B(L, J)
    if model_name == "model_C":
        return compute_idx_model_C(L, J)
    if model_name == "model_Mopt":
        return compute_idx_model_opt(L, J, dj, dl)
    if model_name == "model_Mopt_dn":
        return compute_idx_model_opt_dn(L, J, dj, dl, dn)
    if model_name in ["model_M1", "model_M1_bis", "model_P0", "model_P1"]:
        return compute_idx_model_M1(L, J, dj, dl)
    if model_name == "model_M2":
        return compute_idx_model_M2(L, J, dj, dl)
    if model_name == "model_M3":
        return compute_idx_model_M3(L, J, dj, dl)
    if model_name == "model_M1_bis":
        return compute_idx_model_M1_bis(L, J, dj, dl)
    if model_name == "model_M1_only_mixin":
        return compute_idx_model_M1_only_mixing(L, J, dj, dl)
    if model_name == "model_M1_som":
        return compute_idx_model_M1_strictly_only_mixing(L, J, dj, dl)
    if model_name == "model_M1_no_mixin":
        return compute_idx_model_M1_no_mixing(L, J, dj, dl)
    if model_name == "model_M4":
        return compute_idx_model_M4(L, J, dj, dl)
    if model_name == "model_BL":
        return compute_idx_model_BL(J, dj)
    if model_name == "model_BL_NSF":
        return compute_idx_model_BL_NSF(J, dj)
    if model_name == "model_BL_NSF_GAUSSIAN":
        return compute_idx_model_BL_NSF_Gaussian(J, dj)
    if model_name == "model_BL_NSF_GAUSSIAN_PURE":
        return compute_idx_model_BL_NSF_Gaussian(J, dj)
    if model_name == "model_BL_ONLY_SCALING":
        return compute_idx_model_BL_ONLY_SCALING(J, dj)


def add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, dl, j1, j2, k1, k2):
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = idx_lists
    for ell2 in range(L2):
        if periodic_dis(0, ell2, L2) <= dl:
            idx_j1.append(j1)
            idx_j2.append(j2)
            idx_k1.append(k1)
            idx_k2.append(k2)
            idx_ell2.append(ell2)

def add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, dl, j1, j2, k1, k2, dn1, dn2):
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2 = idx_lists
    for ell2 in range(L2):
        if periodic_dis(0, ell2, L2) <= dl:
            idx_j1.append(j1)
            idx_j2.append(j2)
            idx_k1.append(k1)
            idx_k2.append(k2)
            idx_ell2.append(ell2)
            idx_dn1.append(dn1)
            idx_dn2.append(dn2)

def add_k_and_j_for_BL(idx_lists, j1, j2, k1, k2):
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = idx_lists
    for bl_k_2 in range(3):
        idx_j1.append(j1)
        idx_j2.append(j2)
        idx_k1.append(k1)
        idx_k2.append(k2)
        idx_ell2.append(bl_k_2)

def add_k_and_j_for_BL_NSF(idx_lists, j1, j2, k1, k2):
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = idx_lists
    for bl_k_2 in range(4):
        idx_j1.append(j1)
        idx_j2.append(j2)
        idx_k1.append(k1)
        idx_k2.append(k2)
        idx_ell2.append(bl_k_2)


def get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2):
    idx_wph = dict()
    idx_wph['j1'] = torch.tensor(idx_j1).type(torch.long)
    idx_wph['k1'] = torch.tensor(idx_k1).type(torch.long)
    idx_wph['ell2'] = torch.tensor(idx_ell2).type(torch.long)
    idx_wph['j2'] = torch.tensor(idx_j2).type(torch.long)
    idx_wph['k2'] = torch.tensor(idx_k2).type(torch.long)

    return idx_wph

def get_idx_wph_dn(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2):
    idx_wph = dict()
    idx_wph['j1'] = torch.tensor(idx_j1).type(torch.long)
    idx_wph['k1'] = torch.tensor(idx_k1).type(torch.long)
    idx_wph['ell2'] = torch.tensor(idx_ell2).type(torch.long)
    idx_wph['j2'] = torch.tensor(idx_j2).type(torch.long)
    idx_wph['k2'] = torch.tensor(idx_k2).type(torch.long)
    idx_wph['dn1'] = torch.tensor(idx_dn1).type(torch.long)
    idx_wph['dn2'] = torch.tensor(idx_dn2).type(torch.long)

    return idx_wph

def compute_idx_model_A(L, J, dn, dj, dl):
    """
     Gaussian model
    """
    L2 = L * 2
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2 = [], [], [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2)
    for j1 in range(J):
        add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, dl, j1, j1, 1, 1, 0, 0)
        for n in range(dn):
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 1, 1, 0, (n + 1))
        for j2 in range(j1 + 1, min(j1 + dj + 1, J)):
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, dl, j1, j2, 1, 1, 0, 0)

    return get_idx_wph_dn(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2)


def compute_idx_model_B(L, J):
    """
     model B of Zhang&Mallat 2019
    """
    L2 = L * 2
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)
    for j1 in range(J):
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 0, 0)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 1, 1)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 0, 1)

    return get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)


def compute_idx_model_C(L, J):
    """
     model C of Zhang&Mallat 2019
    """
    L2 = L * 2
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)
    for j1 in range(J):
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 0, 0)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 1, 1)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 0, 1)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 0, 2)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 0, 3)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 1, 2)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 1, 3)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 2, 2)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 2, 3)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j1, 3, 3)

    for j1 in range(J-1):
        j2 = j1+1
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j2, 0, 0)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j2, 0, 1)

        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j2, 1, 0)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j2, 1, 1)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j2, 1, 2)
        add_k_and_j_for_all_ell_in_idx_list(idx_lists, L2, L2//4, j1, j2, 1, 3)

    return get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)



def compute_idx_model_opt_dn(L, J, dj, dl, dn):
    """
    Optimal model ?
    """
    L2 = L * 2
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2 = [], [], [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2)

    # j1=j2, k1=0,1, k2=0 or 1
    for j1 in range(J):
        add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 0, 1, 0, 0)
        add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 1, 1, 0, 0)
        add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, dl, j1, j1, 0, 0, 0, 0)
        if j1 == J - 1:
            max_dn = 0
        elif j1 == J - 2:
            max_dn = min(1, dn)
        else:
            max_dn = dn
        for n in range(4*max_dn):
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 1, 1, 0, (n+1))
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j1, 0, 0, 0, (n+1))


    # k1 = 0,1
    # k2 = 0,1 or 2**(j2-j1)
    # k2 > k1
    # j1+1 <= j2 <= min(j1+dj,J-1)
    for j1 in range(J):
        for j2 in range(j1 + 1, min(j1 + dj + 1, J)):
            if j2 == J - 1:
                max_dn = 0
            elif j2 == J - 2:
                max_dn = min(1, dn)
            else:
                max_dn = dn
            for n in range(4 * max_dn):
                add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j2, 1, 2 ** (j2 - j1), 0, (n+1))
                add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j2, 0, 1, 0, (n + 1))
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, 0, j1, j2, 1, 2**(j2-j1), 0, 0)
            add_k_and_j_and_dn_for_all_ell_in_idx_list(idx_lists, L2, dl, j1, j2, 0, 1, 0, 0)

    print("Total number of coefficient: " + str(len(idx_k2)))

    return get_idx_wph_dn(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2, idx_dn1, idx_dn2)


def compute_idx_model_BL(J, dj):
    """
    Compute a model using BL orthogonal wavelet.
    """
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)

    for j1 in range(J):
        for j2 in range(j1, min(j1 + dj + 1, J)):
            add_k_and_j_for_BL(idx_lists, j1, j2, 0, 0)
            add_k_and_j_for_BL(idx_lists, j1, j2, 1, 1)
            add_k_and_j_for_BL(idx_lists, j1, j2, 1, 2)

    return get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)


def compute_idx_model_BL_NSF(J, dj):
    """
    Compute a model using BL orthogonal wavelet using the non-standard form.
    """
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)

    for j1 in range(J):
        for j2 in range(j1, min(j1 + dj + 1, J)):
            add_k_and_j_for_BL_NSF(idx_lists, j1, j2, 0, 0)
            add_k_and_j_for_BL_NSF(idx_lists, j1, j2, 1, 1)
            add_k_and_j_for_BL_NSF(idx_lists, j1, j2, 1, 2)

    return get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)

def compute_idx_model_BL_NSF_Gaussian(J, dj):
    """
    Compute a model using only scaling functions at each scales ** should not work hopefully**
    """
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)

    for j1 in range(J):
        for j2 in range(j1, min(j1 + dj + 1, J)):
            add_k_and_j_for_BL_NSF(idx_lists, j1, j2, 1, 1)
    return get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)


def compute_idx_model_BL_ONLY_SCALING(J, dj):
    """
    Compute a model using only scaling functions at each scales ** should not work hopefully**
    """
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = [], [], [], [], []
    idx_lists = (idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)

    for j1 in range(J):
        for j2 in range(j1, min(j1 + dj + 1, J)):
            add_k_and_j_for_BL_ONLY_SCALING(idx_lists, j1, j2, 0, 0)
            add_k_and_j_for_BL_ONLY_SCALING(idx_lists, j1, j2, 1, 1)
            add_k_and_j_for_BL_ONLY_SCALING(idx_lists, j1, j2, 1, 2)
    return get_idx_wph(idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2)



def add_k_and_j_for_BL_ONLY_SCALING(idx_lists, j1, j2, k1, k2):
    idx_j1, idx_j2, idx_k1, idx_k2, idx_ell2 = idx_lists
    idx_j1.append(j1)
    idx_j2.append(j2)
    idx_k1.append(k1)
    idx_k2.append(k2)
    idx_ell2.append(0)
