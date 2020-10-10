import config

from circuit import phase_encoder, amplitude_modifier, make_entangle

from torch import Tensor, tensor
from torch import zeros, ones, eye
from torch import stack as tstack
from torch import cat as tcat
from torch import sqrt as tsqrt
from torch import cos, sin
from torch import float32
from torch import einsum

from numpy import array
from numpy import e, pi
from numpy import power, complex
from numpy import sqrt, ceil, log2
from numpy import kron as nkron

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from functools import reduce

from time import time


##


def ID_matrix():

    return ComplexTensor.__graph_copy__(None, tensor([[1.0,.0],[.0,1.0]]), zeros(2,2))

# id_matrix = ID_matrix()


def RX_matrix(angle):
    angle = angle/2

    real = eye(2,2)
    real *= cos(angle)

    imag = zeros(2,2)
    imag[0][1] = -sin(angle)
    imag[1][0] = -sin(angle)

    return ComplexTensor.__graph_copy__(None,real,imag)


def RY_matrix(angle):
    angle = angle/2

    real = eye(2,2)
    real *= cos(angle)
    real[1][0] = sin(angle)
    real[0][1] = -sin(angle)

    imag = zeros(2,2)

    return ComplexTensor.__graph_copy__(None, real, imag)

def RY_matrix_opt(angle):
    angle = angle/2

    real = eye(2,2)
    real *= cos(angle)
    real[1][0] = sin(angle)
    real[0][1] = -sin(angle)

    return real


def RZ_matrix(angle):
    angle = angle/2

    real = eye(2,2)
    real *= cos(angle)

    imag = zeros(2,2)
    imag *= sin(angle)
    imag[0][0] *= -1

    return ComplexTensor.__graph_copy__(None, real, imag)


##


def OP_RX(wave_state, which_ones, which_angles):
    hm_qbits = int(log2(wave_state.shape[0]))
    operation = kronecker([RX_matrix(which_angles[which_ones.index(i)]) if i in which_ones else ID_matrix() for i in range(hm_qbits)])
    return operation.mm(wave_state)

def OP_RY(wave_state, which_ones, which_angles):
    hm_qbits = int(log2(wave_state.shape[0]))
    operation = kronecker([RY_matrix(which_angles[which_ones.index(i)]) if i in which_ones else ID_matrix() for i in range(hm_qbits)])
    return operation.mm(wave_state)

def OP_RZ(wave_state, which_ones, which_angles):
    hm_qbits = int(log2(wave_state.shape[0]))
    operation = kronecker([RZ_matrix(which_angles[which_ones.index(i)]) if i in which_ones else ID_matrix() for i in range(hm_qbits)])
    return operation.mm(wave_state)


##


def zeros_complex(hm_rows, hm_cols):
    return ComplexTensor.__graph_copy__(None,zeros(hm_rows,hm_cols),zeros(hm_rows,hm_cols))


# zero_state = ComplexTensor([[1],
#                             [0],
#
#                             [0],
#                             [0]])

def initialize_wavestate(hm_qbits):
    # return kronecker([zero_state]*hm_qbits)
    real = zeros(2**hm_qbits,1)
    real[0][0] = 1
    imag = zeros(2**hm_qbits,1)
    return ComplexTensor.__graph_copy__(None,real,imag)


def probs(wave_state):
    return wave_state.abs().pow(2)


## this is where shit hits the fan.


    # kroneckers


def k0(a,b):

    return einsum("p,r->pr", a, b).view(a.size(0)*b.size(0))


def k(a,b):

    return einsum("ab,cd->acbd",a,b).view(a.size(0)*b.size(0),a.size(1)*b.size(1))


def kron(a,b):

    return ComplexTensor.__graph_copy__(None, k(a.real,b.real) - k(a.imag,b.imag),
                                              k(a.imag,b.real) + k(a.real,b.imag))


def kronecker(args):
    return reduce(kron,reversed(args))


    # batch kroneckers


def batch_k_rz(a,b):

    return einsum('bp,br->bpr',a,b).view(a.size(0),-1)

def batch_k_rx(a,b): # also y.

    return einsum("kab,kcd->kacbd",a,b).view(a.size(0),a.size(1)*b.size(1),a.size(2)*b.size(2))


    # batch gates


def batch_rz(batch_params):

    batch_params = batch_params /2

    batch_real = None
    batch_imag = None

    for i in range(batch_params.size(1)):

        params_slice = batch_params[:,i:i+1]

        params_real = tcat([tcat([cos(param).view(1,1)]*2, dim=1) for param in params_slice], dim=0)
        params_imag = tcat([tcat([-sin(param).view(1,1),sin(param).view(1,1)], dim=1) for param in params_slice], dim=0)

        batch_real_ = params_real if batch_real is None else batch_k_rz(params_real,batch_real) - batch_k_rz(params_imag,batch_imag)
        batch_imag_ = params_imag if batch_imag is None else batch_k_rz(params_real,batch_imag) + batch_k_rz(params_imag,batch_real)

        batch_real = batch_real_
        batch_imag = batch_imag_

    batch_real = tstack([eye(batch_real.size(1),batch_real.size(1))]*batch_real.size(0),dim=0) \
                * batch_real.view(batch_real.size(0),batch_real.size(1),1)

    batch_imag = tstack([eye(batch_imag.size(1),batch_imag.size(1))]*batch_imag.size(0),dim=0) \
                * batch_imag.view(batch_real.size(0),batch_real.size(1),1)

    return ComplexTensor.__graph_copy__(None,batch_real,batch_imag)

def batch_rz_opt(batch_params):

    batch_params = batch_params /2

    batch_real = None
    batch_imag = None

    for i in range(batch_params.size(1)):

        params_slice = batch_params[:,i:i+1]

        params_real = tcat([ones(1,2)*cos(param) for param in params_slice], dim=0)
        params_imag = tcat([tcat([-sin(param).view(1,1),sin(param).view(1,1)], dim=1) for param in params_slice], dim=0)

        batch_real_ = params_real if batch_real is None else batch_k_rz(params_real,batch_real) - batch_k_rz(params_imag,batch_imag)
        batch_imag_ = params_imag if batch_imag is None else batch_k_rz(params_real,batch_imag) + batch_k_rz(params_imag,batch_real)

        batch_real = batch_real_
        batch_imag = batch_imag_

    hm_params = batch_real.size(1)

    batch_real = tstack([tstack([batch_real[:,rowcol]]*hm_params,dim=1) for rowcol in range(hm_params)],dim=2)
    batch_imag = tstack([tstack([batch_imag[:,rowcol]]*hm_params,dim=1) for rowcol in range(hm_params)],dim=2)

    return ComplexTensor.__graph_copy__(None,batch_real,batch_imag)


def batch_rx(batch_params):

    batch_params = batch_params /2

    batch_real = None
    batch_imag = None

    for i in range(batch_params.size(1)):
        params_slice = batch_params[:,i:i+1]

        params_real = tstack([tcat([tcat([cos(param).view(1,1),tensor(.0).view(1,1)],dim=1),tcat([tensor(.0).view(1,1),cos(param).view(1,1)],dim=1)],dim=0) for param in params_slice],dim=0)
        params_imag = tstack([tcat([tcat([tensor(.0).view(1,1),-sin(param).view(1,1)],dim=1),tcat([-sin(param).view(1,1),tensor(.0).view(1,1)],dim=1)],dim=0) for param in params_slice],dim=0)

        batch_real_ = params_real if batch_real is None else batch_k_rx(params_real,batch_real) - batch_k_rx(params_imag,batch_imag)
        batch_imag_ = params_imag if batch_imag is None else batch_k_rx(params_real,batch_imag) + batch_k_rx(params_imag,batch_real)

        batch_real = batch_real_
        batch_imag = batch_imag_

    return ComplexTensor.__graph_copy__(None, batch_real, batch_imag)


def batch_ry(batch_params):

    batch_params = batch_params /2

    batch_real = None

    for i in range(batch_params.size(1)):

        params_slice = batch_params[:,i:i+1]

        params_real = tstack([tcat([tcat([cos(param).view(1,1), -sin(param).view(1,1)],dim=1),
                                    tcat([sin(param).view(1,1), cos(param).view(1,1)],dim=1)], dim=0) for param in
                              params_slice], dim=0)

        batch_real = params_real if batch_real is None else batch_k_rx(params_real, batch_real)

    return ComplexTensor.__graph_copy__(None, batch_real, zeros(batch_real.size()))

def batch_ry_opt(batch_params):

    batch_params = batch_params /2

    batch_real = None

    for i in range(batch_params.size(1)):

        params_slice = batch_params[:,i:i+1]

        params_real = tstack([tcat([tcat([cos(param).view(1,1), -sin(param).view(1,1)],dim=1),
                                    tcat([sin(param).view(1,1), cos(param).view(1,1)],dim=1)], dim=0) for param in
                              params_slice], dim=0)

        batch_real = params_real if batch_real is None else batch_k_rx(params_real, batch_real)

    return batch_real


def batch_u3(batch_params):

    batch_real = None
    batch_imag = None

    for i in range(batch_params.size(1) // 3):
        params_slice = batch_params[:,i*3:(i+1)*3]

        params_real = tstack([tcat([tcat([cos(p0/2).view(1,1), -(cos(p2)*sin(p0/2)).view(1,1)], dim=1),
                                    tcat([(cos(p1)*sin(p0/2)).view(1,1), (cos(p1+p2)*cos(p0/2)).view(1,1)], dim=1)], dim=0) for p0,p1,p2 in
                              params_slice], dim=0)
        params_imag = tstack([tcat([tcat([tensor(.0).view(1,1), -(sin(p2)*sin(p0/2)).view(1,1)], dim=1),
                                    tcat([(sin(p1)*sin(p0/2)).view(1,1), (sin(p1+p2)*cos(p0/2)).view(1,1)], dim=1)], dim=0) for p0,p1,p2 in
                              params_slice], dim=0)

        batch_real_ = params_real if batch_real is None else batch_k_rx(params_real, batch_real) - batch_k_rx(params_imag, batch_imag)
        batch_imag_ = params_imag if batch_imag is None else batch_k_rx(params_real, batch_imag) + batch_k_rx(params_imag, batch_real)

        batch_real = batch_real_
        batch_imag = batch_imag_

    return ComplexTensor.__graph_copy__(None, batch_real, batch_imag)


    ## ansatz zero


cx_guide = {}
a0_circuit = []

from copy import deepcopy

def circuit_a0():

    global cx_guide
    global a0_circuit

    if not cx_guide:
        for id_control in reversed(range(config.hm_qbits)):
            cx_guide[id_control] = {}
            for id_target in range(id_control):
                c = QuantumCircuit(config.hm_qbits, config.hm_qbits)
                c.cx(id_control, id_target)
                cx_guide[id_control][id_target] = tensor(Operator(c).data.real,dtype=float32)

    if not a0_circuit:
        a0_circuit = amplitude_modifier()

    return cx_guide, a0_circuit


def batch_a0(batch_params):

    batch_params = batch_params /2

    cx_guide, circuit = circuit_a0()

    batch_real = tstack([eye(config.statevec_size,config.statevec_size)]*batch_params.size(0),dim=0)

    param_ctr = 0

    for _ in range(config.circuit_layers):

        for element in circuit:

            if element[0] == 'RY':

                batch_real = ry_a0(batch_params[:,param_ctr],element[1]) @ batch_real
                param_ctr += 1

            elif element[0] == 'CX':

                batch_real = cx_guide[element[1]][element[2]] @ batch_real

    return batch_real


def ry_a0(params_slice, qbit_id):

    batch_size = params_slice.size(0)
    sector_size = 2**qbit_id
    group_size = sector_size*2

    params_slice = params_slice.view(batch_size,1,1)

    group = cos(params_slice) * tstack([eye(group_size,group_size)]*batch_size,dim=0)

    sector_leftdown = sin(params_slice) * tstack([eye(sector_size,sector_size)]*batch_size,dim=0)
    sector_rightup = -sector_leftdown

    group[:,sector_size:,:sector_size] = sector_leftdown
    group[:,:sector_size,sector_size:] = sector_rightup

    hm_repeats = config.statevec_size//group_size

    matrix = zeros(batch_size,config.statevec_size,config.statevec_size)

    for i in range(hm_repeats):
        f = i*group_size
        t = (i+1)*group_size

        matrix[:,f:t,f:t] = group

    return matrix


##


def encoder(timestep,hm_qbits):

    c = QuantumCircuit(hm_qbits,hm_qbits)

    phase_encoder(c,timestep)

    return ComplexTensor(Operator(c).data)


def ansatz(params):

    params = params * 2*pi

    if config.ansatz_mode == 0:

        return batch_diagonalize_matrices_real(batch_a0(params))

    elif config.ansatz_mode == 1:

        # return batch_diagonalize_matrices_real(batch_ry(params))
        # rzs = batch_rz_opt(params[:,:config.hm_qbits])
        # rys = batch_ry_opt(params[:,config.hm_qbits:])
        # return batch_diagonalize_matrices(rzs*rys)

        return batch_diagonalize_matrices(batch_u3(params))


def entangler(hm_qbits):

    c = QuantumCircuit(hm_qbits,hm_qbits)

    make_entangle(c)

    return ComplexTensor(Operator(c).data)


##


def prop_circuits(params, timesteps):

    batch_size = params.size(0)
    timestep_size = timesteps.size(1)

    hm_qbits = int(ceil(log2(timestep_size)))

    #start = time()

    if config.reconstruct_qstate:
        state_vectors = tcat([tsqrt(timesteps),zeros(batch_size,2**hm_qbits-timestep_size)],dim=1).view(2**hm_qbits*batch_size,1)
    else:
        state_vec = zeros(2**hm_qbits,1) ; state_vec[0][0] = 1
        state_vectors = tcat([state_vec]*batch_size,dim=0)

    states = state_vectors # states = ComplexTensor.__graph_copy__(None, state_vectors, zeros(state_vectors.size()))
    # no need to complexify.

    #enct = time() - start

    #start = time()

    ansatzs = ansatz(params)

    #anst = time() - start

    #start = time()

    states = ansatzs.mm(states)

    if config.entangle_mode != -1:

        entanglings = diagonalize_matrices([entangler_matrix]*batch_size)
        states = entanglings.mm(states)

    #entt = time() - start

    #print(f'>> encoding took {enct}, ansatz took {anst}, entangling took {entt}')

    return probs(states).view(batch_size,pow(2,hm_qbits))


##


def diagonalize_matrices(matrices):

    hm_matrices = len(matrices)
    single_matrix_size = matrices[0].size()[0]

    real = zeros(hm_matrices*single_matrix_size,hm_matrices*single_matrix_size)
    imag = zeros(hm_matrices*single_matrix_size,hm_matrices*single_matrix_size)

    for i in range(hm_matrices):
        real[i*single_matrix_size:(i+1)*single_matrix_size,i*single_matrix_size:(i+1)*single_matrix_size] = matrices[i].real
        imag[i*single_matrix_size:(i+1)*single_matrix_size,i*single_matrix_size:(i+1)*single_matrix_size] = matrices[i].imag

    return ComplexTensor.__graph_copy__(None,real,imag)


def batch_diagonalize_matrices(matrices):

    hm_matrices = matrices.real.size(0)
    single_matrix_size = matrices.real.size(1)

    real = zeros(hm_matrices*single_matrix_size,hm_matrices*single_matrix_size)
    imag = zeros(hm_matrices*single_matrix_size,hm_matrices*single_matrix_size)

    for i in range(hm_matrices):
        real[i*single_matrix_size:(i+1)*single_matrix_size,i*single_matrix_size:(i+1)*single_matrix_size] = matrices.real[i]
        imag[i*single_matrix_size:(i+1)*single_matrix_size,i*single_matrix_size:(i+1)*single_matrix_size] = matrices.imag[i]

    return ComplexTensor.__graph_copy__(None,real,imag)

def batch_diagonalize_matrices_real(matrices):

    hm_matrices = matrices.size(0)
    single_matrix_size = matrices.size(1)

    real = zeros(hm_matrices*single_matrix_size,hm_matrices*single_matrix_size)

    for i in range(hm_matrices):
        real[i*single_matrix_size:(i+1)*single_matrix_size,i*single_matrix_size:(i+1)*single_matrix_size] = matrices[i]

    return real


##



## torch complex (modded) ##

import numpy as np
import torch
import re

class ComplexTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        if isinstance(x, np.ndarray) and 'complex' in str(x.dtype):
            r = x.real
            i = x.imag
            x = np.concatenate([r, i], axis=0)
        if type(x) is int and len(args) == 1:
            x = x * 2
        elif len(args) >= 2:
            size_args = list(args)
            size_args[0] *= 2
            args = tuple(size_args)
        else:
            if isinstance(x, torch.Tensor):
                s = x.size()[0]
            elif isinstance(x, list):
                s = len(x)
            elif isinstance(x, np.ndarray):
                s = x.shape[0]
            if not (s % 2 == 0): raise Exception('0th dim must be even. ComplexTensor is 2 real matrices under the hood')
        new_t = super().__new__(cls, x, *args, **kwargs)
        return new_t

    def __deepcopy__(self, memo):
        if not self.is_leaf:
            raise RuntimeError("Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment")
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            if self.is_sparse:
                new_tensor = self.clone()
                new_tensor.__class__ = ComplexTensor
            else:
                new_storage = self.storage().__deepcopy__(memo)
                new_tensor = self.new()
                new_tensor.__class__ = ComplexTensor
                new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
            memo[id(self)] = new_tensor
            new_tensor.requires_grad = self.requires_grad
            return new_tensor

    @property
    def real(self):
        self.realimag_calling = 1
        return self[:self.size()[0], ...]

    @property
    def imag(self):
        self.realimag_calling = 1
        return self[self.size()[0]:, ...]

    def __graph_copy__(self, real, imag):
        result = torch.cat([real, imag], dim=0)
        result.__class__ = ComplexTensor
        return result

    def __graph_copy_scalar__(self, real, imag):
        result = torch.stack([real, imag], dim=0)
        result.__class__ = ComplexScalar
        return result

    def __add__(self, other):
        real = self.real
        imag = self.imag
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            real = real + other
        elif type(other) is ComplexTensor:
            real = real + other.real
            imag = imag + other.imag
        elif np.isreal(other):
            real = real + other
        else:
            real = real + other.real
            imag = imag + other.imag
        return self.__graph_copy__(real, imag)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        real = self.real
        imag = self.imag
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            real = real - other
        elif type(other) is ComplexTensor:
            real = real - other.real
            imag = imag - other.imag
        elif np.isreal(other):
            real = real - other
        else:
            real = real - other.real
            imag = imag - other.imag
        return self.__graph_copy__(real, imag)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        real = self.real.clone()
        imag = self.imag.clone()
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            real = real * other
            imag = imag * other
        elif type(other) is ComplexTensor:
            ac = real * other.real
            bd = imag * other.imag
            ad = real * other.imag
            bc = imag * other.real
            real = ac - bd
            imag = ad + bc
        elif np.isreal(other):
            real = real * other
            imag = imag * other
        else:
            ac = real * other.real
            bd = imag * other.imag
            ad = real * other.imag
            bc = imag * other.real
            real = ac - bd
            imag = ad + bc
        return self.__graph_copy__(real, imag)

    def __truediv__(self, other):
        real = self.real.clone()
        imag = self.imag.clone()
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            raise NotImplementedError
        elif type(other) is ComplexTensor:
            raise NotImplementedError
        elif np.isreal(other):
            real = real / other
            imag = imag / other
        else:
            raise NotImplementedError
        return self.__graph_copy__(real, imag)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def mm(self, other):
        real = self.real.clone()
        imag = self.imag.clone()
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            real = real.mm(other)
            imag = imag.mm(other)
        elif type(other) is ComplexTensor:
            ac = real.mm(other.real)
            bd = imag.mm(other.imag)
            ad = real.mm(other.imag)
            bc = imag.mm(other.real)
            real = ac - bd
            imag = ad + bc
        return self.__graph_copy__(real, imag)

    def t(self):
        real = self.real.t()
        imag = self.imag.t()
        return self.__graph_copy__(real, imag)

    def abs(self):
        result = torch.sqrt(self.real**2 + self.imag**2)
        return result

    def sum(self, *args):
        real_sum = self.real.sum(*args)
        imag_sum = self.imag.sum(*args)
        return ComplexScalar(real_sum, imag_sum)

    def mean(self, *args):
        real_mean = self.real.mean(*args)
        imag_mean = self.imag.mean(*args)
        return ComplexScalar(real_mean, imag_mean)

    @property
    def grad(self):
        g = self._grad
        g.__class__ = ComplexGrad
        return g

    def cuda(self):
        real = self.real.cuda()
        imag = self.imag.cuda()
        return self.__graph_copy__(real, imag)

    def __repr__(self):
        real = self.real.flatten()
        imag = self.imag.flatten()
        strings = np.asarray([complex(a,b) for a, b in zip(real, imag)]).astype(np.complex64)
        strings = strings.__repr__()
        strings = re.sub('array', 'tensor', strings)
        return strings

    def __str__(self):
        return self.__repr__()

    def is_complex(self):
        return True

    def size(self, *args):
        size = self.data.size(*args)
        size = list(size)
        size[0] //= 2
        size = torch.Size(size)
        return size

    @property
    def shape(self):
        size = self.data.shape
        size = list(size)
        size[0] //= 2
        size = torch.Size(size)
        return size

    def __getitem__(self, item):
        if hasattr(self,'realimag_calling'):
            if self.realimag_calling:
                self.realimag_calling = None
                return super(ComplexTensor, self).__getitem__(item)
        return self.__graph_copy__(self.real[item], self.imag[item])

class ComplexGrad(torch.Tensor):

    def __deepcopy__(self, memo):
        if not self.is_leaf:
            raise RuntimeError("Only Tensors created explicitly by the user "
                               "(graph leaves) support the deepcopy protocol at the moment")
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            if self.is_sparse:
                new_tensor = self.clone()
                new_tensor.__class__ = ComplexGrad
            else:
                new_storage = self.storage().__deepcopy__(memo)
                new_tensor = self.new()
                new_tensor.__class__ = ComplexGrad
                new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
            memo[id(self)] = new_tensor
            new_tensor.requires_grad = self.requires_grad
            return new_tensor

    def __repr__(self):
        size = self.size()
        split_i = size[0] // 2
        real = self[:split_i]
        imag = self[split_i:]
        size_r = real.size()

        real = real.view(-1)
        imag = imag.view(-1)

        strings = np.asarray([f'({a}{"+" if b > 0 else "-"}{abs(b)}j)' for a, b in zip(real, imag)])
        strings = strings.reshape(*size_r)
        strings = f'tensor({strings.__str__()})'
        strings = re.sub('\n', ',\n       ', strings)
        return strings

    def __str__(self):
        return self.__repr__()

class ComplexScalar(object):

    def __init__(self, real, imag):
        self._real = real
        self._imag = imag

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    def backward(self):
        self._real.backward()

    def __repr__(self):
        return str(complex(self.real.item(), self.imag.item()))

    def __str__(self):
        return self.__repr__()

def __graph_copy__(real, imag):
    result = torch.cat([real, imag], dim=-2)
    result.__class__ = ComplexTensor
    return result

def __apply_fx_to_parts(items, fx, *args, **kwargs):
    r = [x.real for x in items]
    r = fx(r, *args, **kwargs)

    i = [x.imag for x in items]
    i = fx(i, *args, **kwargs)

    return __graph_copy__(r, i)

def stack(items, *args, **kwargs):
    return __apply_fx_to_parts(items, torch.stack, *args, **kwargs)

def cat(items, *args, **kwargs):
    return __apply_fx_to_parts(items, torch.cat, *args, **kwargs)



## final touch ##

entangler_matrix = entangler(config.hm_qbits)