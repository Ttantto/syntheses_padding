from .backend import fft, conjugate


def compute_padding(M, N, J):
    """
         Precomputes the future padded size.

         Parameters
         ----------
         M, N : int
             input size

         Returns
         -------
         M, N : int
             padded size
    """
    M_padded = ((M + 2 ** J) // 2 ** J + 1) * 2 ** J
    N_padded = ((N + 2 ** J) // 2 ** J + 1) * 2 ** J
    return M_padded, N_padded


def fft2_c2c(x):
    return fft(x)


def ifft2_c2c(x):
    return fft(x, inverse=True)


def periodic_dis(i1, i2, per):
    if i2 > i1:
        return min(i2-i1, i1-i2+per)
    else:
        return min(i1-i2, i2-i1+per)


def periodic_signed_dis(i1, i2, per):
    if i2 < i1:
        return i2 - i1 + per
    else:
        return i2 - i1


def check_symmetry_due_to_real_field(xpsi_bc):
    """

    Args:
        xpsi_bc: (J, L2, M, N, 2)

    Returns:

    """
    L2 = xpsi_bc.shape[1]
    diff = xpsi_bc[:, :L2//2, ...] - conjugate(xpsi_bc[:, L2//2:, :, :, :])

    print("xpsi max:" + str(xpsi_bc.max()))
    print("xpsi mean:" + str(xpsi_bc.abs().mean()))
    print("Diff max:" +str(diff.max()))
    print("Diff mean:" + str(diff.abs().mean()))
