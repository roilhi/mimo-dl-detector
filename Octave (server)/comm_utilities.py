import numpy as np
from numpy.random import standard_normal
from itertools import product
'''
class comm_utilities:
    def __init__(self,M,Nt,Nr,No):
        self.M = M
        self.Nt = Nt
        self.Nr = Nr
        self.No = No
'''
def qam_symbol_generator(M):
    '''
    Generates QAM constellation symbols.
    Input --> M: QAM constellation order
    Output ---> M-QAM alphabet
    '''
    N = np.log2(M)
    if N != np.round(N):
        raise ValueError("M must be 2^n for n=0,1,2...")
    m = np.arange(M)
    c = np.sqrt(M)
    b = -2*(np.array(m)%c) + c-1
    a = 2*np.floor(np.array(m) / c ) - c+1
    s = list((a+1j*b)) #list of QAM symbols
    return s
def H_channel(Nr,Nt):
    '''
    Generates the H channel matrix having:
    Inputs
    `N_t` --> number of Tx antennas
    Nr ---> number of Rx antennas
    Output: H --> channel matrix of complex coefficients
    '''
    return (1/np.sqrt(2))*(standard_normal((Nr,Nt))+1j*(standard_normal((Nr,Nt))))
def awgn_noise(No, Nr):
    '''
    Generates AWGN additive noise:
    Inputs:
    Nr ---> Number of Rx antennas
    No --> Noise spectrum density No = Es / ((10**(EbNo/10))*np.log2(M))
    Output:
    vector of AWGN noise
    '''
    return np.sqrt(No/2)*(standard_normal((Nr,1))+1j*standard_normal((Nr,1)))
def biterr_calculation(mod_idx, demod_idx, bit_width):
    '''
    Calculates the number of error bits (or different) having 2 lists with the
    indices of the transmitted and detected QAM symbols.
    Inputs:
    mod_bits --> list with the indices of the transmitted symbols
    demod_bits --> list with the indices of the decoded symbols
    bit_width --> Number of bits to be used for conversion
    Outputs:
    bit_err --> the number of bits which are different from the indices lists
    pe --> the probability of bit errors
    '''
    if isinstance(mod_idx,list) and isinstance(demod_idx,list):
        if len(mod_idx) != len(demod_idx):
            raise ValueError("The lists of indices must have the same lenght")
        mod_bits = ''.join([np.binary_repr(dec, width=bit_width) for dec in mod_idx])
        demod_bits = ''.join([np.binary_repr(dec, width=bit_width) for dec in demod_idx])
    else:
        mod_bits = np.binary_repr(mod_idx, width=bit_width)
        demod_bits = np.binary_repr(demod_idx, width=bit_width)
        bit_err = sum(1 for a, b in zip(mod_bits, demod_bits) if a != b)
    return bit_err
def qam_mimo_tx_combinations(Nt, qam_constellation_array):
    '''
    Creates an array which simulates all the possible combinations
    of QAM symbols having Nt transmission antennas.
    Inputs: Nt --> Number of transmission antennas
            qam_constellation_array: array of qam symbols (produced by qam symbol generator)
    Output: Array given by the cartesian product of all Nt possible combinations of QAM symbols
    '''
    return np.array(list(product(qam_constellation_array,repeat=Nt)))

def eqsm_constellation(M,qam_sym,Nt,K):
    '''
    This code is based on the E-QSM modulation algorithm described in the paper:
    Castillo-Soria, F. R., Cortez, J., Gutiérrez, C. A., Luna-Rivera, M., & Garcia-Barrientos, A. (2019). 
    Extended quadrature spatial modulation for MIMO wireless communications. 
    Physical Communication, 32, 88-95.

    The output is an extended QSM constellation (EQSM) based on QAM symbols. 
    Each ak sequence is modulated by QSM, then sequences are combined
    Inputs:
            M --> QAM modulation order
            Nt --> Number of Tx antennas
            qam_symb --> qam_symbols for modulation (use qam_symbol_generator function)
            K ---> number of EQSM blocks
    Output: EQSM constellation produced for Nt antennas
    '''
    QSM_constellation = []
    first_LSBs = int(np.log2(M)) #number of LSBs to modulate QAM symbol
    next_LSBs = int(np.log2(Nt)) # number of remaining LSBs to make antena indices
    bits_frame =  first_LSBs + 2*next_LSBs # number of bits in each QSM block ak
    sequences = [np.binary_repr(s,width=M) for s in range(2**(bits_frame))] # possible binary sequences or ak bit blocks
    for sequence in sequences:
        idx_qk_symbol = int(sequence[-first_LSBs:],2) # taking 1st log2(M) bits to modulate QAM
        qk_symbol = qam_sym[idx_qk_symbol]
        xR = np.real(qk_symbol)
        xI = np.imag(qk_symbol) #taking Re{} Im{} parts of QAM symbol
        idx_bits = sequence[:-first_LSBs] #remaining bits to be taken as antena indices
        x1 = np.zeros((Nt,)) #vector to place the real part of symbol
        x2 = np.zeros((Nt,)) # vector to place imaginary part of the symbol
        idx_imag = int(idx_bits[:-next_LSBs],2) #taking MSB index to place the real part of symbol
        idx_real = int(idx_bits[-next_LSBs:],2) # taking next LSB index to place the imaginary part
        x1[idx_real] = xR #placing real part in real index
        x2[idx_imag] = xI  #plaging imaginary part in imaginary index
        QSM_constellation.append(x1+1j*x2) #saving each element in constellation
    W2 = 0.5 # best choice for W2, W1 is ok to be 1
    EQSM_all = list(product(QSM_constellation, repeat=K))
    EQSM_comb = [element[0] + W2*element[1] for element in EQSM_all] # making W1*s1 + W2*s2
    return EQSM_comb