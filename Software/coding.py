#######################################################
#     Spiros Daskalakis                               #
#     last Revision: 17/8/2020                        #
#     Python Version:  3.8.4                          #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################

import params
import numpy as np


def bool2int(x):
    y = 0
    for i, j in enumerate(x):
        if j: y += int(j) << i
    return y


def vec_bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[..., bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret


def produce_G_matrix(CR):
    if CR == 1:  # simple parity check
        # P = np.ones((4, 1), dtype=int)
        # eye_m = np.eye(4, dtype=int)
        # G = np.concatenate((eye_m, P), axis=-1)
        G = np.array([[1, 0, 0, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 1]])
    elif CR == 2:  # Hamming(6, 4) # shortened Hamming
        G = np.array([[1, 0, 0, 0, 0, 1],
                      [0, 1, 0, 0, 1, 0],
                      [0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 1]])
    elif CR == 3:  # Hamming(7, 4)
        G = np.array([[1, 0, 0, 0, 0, 1, 1],
                      [0, 1, 0, 0, 1, 0, 1],
                      [0, 0, 1, 0, 1, 1, 0],
                      [0, 0, 0, 1, 1, 1, 1]])

    elif CR == 4:  # Extended Hamming(8, 4)
        G = np.array([[1, 0, 0, 0, 0, 1, 1, 1],
                      [0, 1, 0, 0, 1, 0, 1, 1],
                      [0, 0, 1, 0, 1, 1, 0, 1],
                      [0, 0, 0, 1, 1, 1, 1, 0]])
    else:
        print("Wrong CR")
    return G


def produce_H_matrix(CR):
    if CR == 1:  # simple parity check
        H = np.array([[1],
                      [1],
                      [1],
                      [1]])
    elif CR == 2:  # Hamming(6, 4) # shortened Hamming
        H = np.array([[0, 1],
                      [1, 0],
                      [1, 1],
                      [1, 1],
                      [1, 0],
                      [0, 1]])
    elif CR == 3:  # Hamming(7, 4)
        H = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0],
                      [1, 1, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

    elif CR == 4:  # Extended Hamming(8, 4)
        H = np.array([[0, 1, 1, 1],
                      [1, 0, 1, 1],
                      [1, 1, 0, 1],
                      [1, 1, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    else:
        print("Wrong CR")
    return H


def Hamming_code(in_vector, G, CR):
    aux = np.zeros(4 + CR, dtype=np.int)
    for i in range(0, 4 + CR):
        sum_a = 0
        for j in range(0, 4):
            sum_a = sum_a + (in_vector[j] * G[j][i])
        aux[i] = int(sum_a % 2)
    return aux


def Sindrome(CR):
    H = produce_H_matrix(CR)
    cl = np.zeros((2 ** params.CR, params.CR + 4), dtype=np.int32)
    cl_found = np.zeros(2 ** params.CR - 1, dtype=np.int32)
    for i in range(0, params.CR + 4):
        # print(2**np.arange(params.CR - 1, -1, -1))
        syn = np.dot(H[i, :], (2 ** np.arange(params.CR - 1, -1, -1))) - 1
        if not cl_found[syn]:
            cl[syn, i] = 1
            cl_found[syn] = 1
        else:
            cl_found[syn] = 2

    if any(cl_found == 0):
        for i1 in range(0, params.CR + 4 - 1):
            for i2 in range(i1 + 1, params.CR + 4):
                syn = np.dot(((H[i1, :] + H[i2, :]) % 2), (2 ** np.arange(params.CR - 1, -1, -1))) - 1
                if not cl_found[syn]:
                    cl[syn, [i1, i2]] = 1
                    cl_found[syn] = 1
                else:
                    cl_found[syn] = 2
    return cl, cl_found


def Hamming_decode(C_est, N_bits, N_codewords, CR):
    H = produce_H_matrix(CR)
    cl, cl_found = Sindrome(CR)

    bits_est = np.zeros(N_bits, dtype=np.int)

    for i in range(0, N_codewords):
        temp = C_est[i, :]

        if CR > 2:

            syn = np.dot(np.dot(temp, H) % 2, (2 ** np.arange(params.CR - 1, -1, -1)))

            if syn != 0 and cl_found[syn] == 1:
                temp = (temp + cl[syn, :]) % 2
        bits_est[i * 4 + np.arange(0, 4)] = temp[0:4]
    return bits_est


def bit_stream_code(bit_vector, N_of_codewords, G, CR):
    res = np.zeros((int(N_of_codewords), 4 + CR), dtype=np.int)
    a = 0
    for s in range(0, bit_vector.size, 4):
        bit_vec = bit_vector[s:s + 4]
        res[a:] = Hamming_code(bit_vec, G, CR)
        a = a + 1
    return res


def inter_liver(C, N_coded_bits, CR, SF):
    inter_leaver_size = (4 + CR) * SF
    N_blocks = N_coded_bits / inter_leaver_size
    N_syms = N_coded_bits / SF
    S = np.zeros((int(N_syms), SF), dtype=np.int)

    for i in range(0, int(N_blocks)):
        for k in range(0, 4 + CR):
            for m in range(0, SF):
                S[(4 + CR) * i + k, m] = C[SF * i + (m - k) % SF, k]
    return S


def de_inter_liver(S_est, N_codewords, N_coded_bits, CR, SF):
    inter_leaver_size = (4 + CR) * SF
    N_blocks = N_coded_bits / inter_leaver_size
    C_est = np.zeros((int(N_codewords), 4 + CR), dtype=np.int)

    for i in range(0, int(N_blocks)):
        for k in range(0, 4 + CR):
            for m in range(0, SF):
                C_est[SF * i + (m - k) % SF, k] = S_est[(4 + CR) * i + k, m]
    return C_est
