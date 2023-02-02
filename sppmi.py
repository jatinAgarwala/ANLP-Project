import numpy as np
import data
from scipy.sparse import csr_matrix, coo_matrix
import torch

def get_sppmi(coocc, smoothing_parameter = 0.75, k_shift = 1.0):
    """
    Function to calculate the SPPMI (Shifted Positive Pointwise Mutual Information)
    Inputs:
        Cooccurrence matric
        Smoothening parameter (alpha)
        k_shift (s)

    Output:
        The corresponding SPPMI
    Since PPMI is biased towards infrequent words and assigns them a higher value, we smoothen the context probabilities.
    """
    total_word_coocc = coocc.sum(axis = 1)
    coocc_smooth = (np.power(total_word_coocc, smoothing_parameter)).reshape((1,-1))    # corrected
    coocc_smooth_sum = coocc_smooth.sum()
    sppmi_mat = coocc.multiply(1.0/ total_word_coocc) * coocc_smooth_sum
    sppmi_mat = sppmi_mat.multiply(1.0/ coocc_smooth)
    sppmi_mat = sppmi_mat._with_data(np.log(sppmi_mat.data), copy=False)

    sppmi_mat.data -= np.log(k_shift)
    # if (k_shift > 1.5):               # the original paper does this only if k_shift > 1.5
        # sppmi_mat.data -= np.log(k_shift)
    sppmi_mat.data = np.clip(sppmi_mat.data, a_min=0.0, a_max=None)
    sppmi_mat = coo_matrix(sppmi_mat)

    values = sppmi_mat.data
    indices = np.vstack((sppmi_mat.row, sppmi_mat.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    return torch.sparse.FloatTensor(i, v, torch.Size(sppmi_mat.shape)).to_dense()