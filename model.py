import config

from ext import pickle_save, pickle_load
from simu import prop_circuits

from torch import tensor, Tensor, cat, stack
from torch import zeros, ones, eye, randn
from torch import sigmoid, tanh, relu, softmax
from torch import pow, log, sqrt, norm
from torch import float32, no_grad
from torch.nn.init import xavier_normal_

from collections import namedtuple

from math import sqrt as psqrt


##


# FF = namedtuple('FF', 'w b')
FF1 = namedtuple('FF1', 'w')
FF2 = namedtuple('FF2', 'w')
#LSTM = namedtuple('LSTM', 'wf bf wk bk wi bi ws bs')
LSTM = namedtuple('LSTM', 'wf wk wi ws')
#GRU = namedtuple('GRU', 'wk bk wa ba wi bi')
GRU = namedtuple('GRU', 'wk wa wi')
IRNN = namedtuple('IRNN', 'wo bo ws bs')


def make_Llayer(in_size, layer_size):

    layer = LSTM(
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        #zeros(1,                  layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        #zeros(1,                  layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        #zeros(1,                  layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        #zeros(1,                  layer_size, requires_grad=True, dtype=float32),
    )

    with no_grad():
        for k,v in layer._asdict().items():
            if k == 'bf':
                v += config.forget_bias
        # layer.bf += config.forget_bias

    if config.init_xavier:
        xavier_normal_(layer.wf)
        xavier_normal_(layer.wk)
        xavier_normal_(layer.ws)
        xavier_normal_(layer.wi, gain=5/3)

    return layer

def make_Glayer(in_size, layer_size):

    layer = GRU(
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        #zeros(1,                  layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        #zeros(1,                  layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        #zeros(1,                  layer_size, requires_grad=True, dtype=float32),
    )

    if config.init_xavier:
        xavier_normal_(layer.wk)
        xavier_normal_(layer.wa)
        xavier_normal_(layer.wi, gain=5/3)

    return layer

def make_Ilayer(in_size, layer_size):

    wo = randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32)
    ws = randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32)

    if config.init_xavier:
        xavier_normal_(wo, gain=psqrt(2))
        xavier_normal_(ws, gain=psqrt(2))

    with no_grad():
        wo[-layer_size:,:] = eye(layer_size, layer_size, requires_grad=True, dtype=float32)
        ws[-layer_size:,:] = eye(layer_size, layer_size, requires_grad=True, dtype=float32)

    layer = IRNN(
        wo,
        zeros(1, layer_size, requires_grad=True, dtype=float32),
        ws,
        zeros(1, layer_size, requires_grad=True, dtype=float32),
    )

    return layer

def make_Flayer1(in_size, layer_size):

    layer = FF1(
        randn(in_size, layer_size, requires_grad=True, dtype=float32),
        # zeros(1, layer_size,       requires_grad=True, dtype=float32),
    )

    return layer

def make_Flayer2(in_size, layer_size):

    layer = FF2(
        randn(in_size, layer_size, requires_grad=True, dtype=float32),
        # zeros(1, layer_size,       requires_grad=True, dtype=float32),
    )

    if config.init_xavier:
        xavier_normal_(layer.w, gain=5/3)

    return layer

make_layer = {
    'l': make_Llayer,
    'g': make_Glayer,
    'i': make_Ilayer,
    'f1': make_Flayer1,
    'f2': make_Flayer2,
}


def prop_Llayer(layer, state, input):

    layer_size = layer.wf.size(1)
    prev_out = state[:,:layer_size]
    state = state[:,layer_size:]

    inp = cat([input,prev_out],dim=1)

    forget = sigmoid(inp@layer.wf)# + layer.bf)
    keep   = sigmoid(inp@layer.wk)# + layer.bk)
    interm = tanh   (inp@layer.wi)# + layer.bi)
    show   = sigmoid(inp@layer.ws)# + layer.bs)

    state = forget*state + keep*interm
    out = show*tanh(state)

    return out, cat([out,state],dim=1)

def prop_Llayer2(layer, state, input):

    inp = cat([input,state],dim=1)

    forget = sigmoid(inp@layer.wf + layer.bf)
    keep   = sigmoid(inp@layer.wk + layer.bk)
    interm = tanh   (inp@layer.wi + layer.bi)
    show   = sigmoid(inp @ layer.ws + layer.bs)

    state  = forget*state + keep*interm
    # inp = cat([input,state],dim=1)

    out = show*tanh(state)

    return out, state

def prop_Glayer(layer, state, input):

    inp = cat([input,state],dim=1)

    keep   = sigmoid(inp@layer.wk)# + layer.bk)
    attend = sigmoid(inp@layer.wa)# + layer.ba)
    interm = tanh(cat([input,attend*state],dim=1)@layer.wi)# + layer.bi)

    state = keep*interm + (1-keep)*state
    out = state

    return out, state

def prop_Ilayer(layer, state, input):

    inp = cat([input,state],dim=1)

    out = relu(inp@layer.wo + layer.bo)
    state = relu(inp@layer.ws + layer.bs)

    return out, state

def prop_Flayer1(layer, inp):

    return softmax(inp@layer.w, dim=1) # + layer.b)

def prop_Flayer2(layer, inp):

    return tanh(inp @ layer.w)

prop_layer = {
    LSTM: prop_Llayer2,
    GRU: prop_Glayer,
    IRNN: prop_Ilayer,
    FF1: prop_Flayer1,
    FF2: prop_Flayer2,
}


def make_model(info=None):

    if not info: info = config.creation_info

    layer_sizes = [e for e in info if type(e)==int]
    layer_types = [e for e in info if type(e)==str]

    return [make_layer[layer_type](layer_sizes[i], layer_sizes[i+1]) for i,layer_type in enumerate(layer_types)]


def prop_model_nocircuit(model, states, inp):
    new_states = []

    out = inp

    state_ctr = 0

    for layer in model:

        if type(layer) != FF1 and type(layer) != FF2:

            out, state = prop_layer[type(layer)](layer, states[state_ctr], out)
            new_states.append(state)
            state_ctr += 1

        else:

            out = prop_layer[type(layer)](layer, out)

        # dropout(out, inplace=True)

    return out, new_states


def prop_model(model, states, inp):

    out, new_states = prop_model_nocircuit(model, states, inp)

    if not config.act_classical_rnn:
        out = prop_circuits(out, inp)

    return out, new_states


def respond_to(model, sequence, states=None):  # , wave_state=None):

    if not states:
        states = empty_states(model)
    # if not wave_state:
    #     wave_state = zeros(1, statevec_size)

    response = []

    teach = int(config.hm_bars_grouped/config.hm_bars_teacher * len(sequence))

    for timestep in sequence[:teach]:

        out, states = prop_model(model,states,timestep) # cat([timestep,wave_state],1)) # wave_state = out.view(1,out.size(0)).float()

        response.append(out)

    for _ in sequence[teach:]:

        out, states = prop_model(model,states,out[:config.timestep_size])

        response.append(out)

    return response, states


def sequence_loss(label, output, do_stack=True):

    if do_stack:
        label = stack(label,dim=0)
        output = stack(output,dim=0)

    if config.loss_squared:
        loss = pow(label-output,2).sum()
    else:
        loss = (label-output).abs().sum()

    return loss


def sgd(model, lr=None, batch_size=None):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    with no_grad():

        for layer in model:
            for param in layer._asdict().values():
                if param.requires_grad:

                    param.grad /=batch_size

                    if config.gradient_clip:
                        param.grad.clamp(min=-config.gradient_clip,max=config.gradient_clip)

                    param -= lr * param.grad
                    param.grad = None


##


moments, variances = [], []


def adaptive_sgd(model, epoch_nr, lr=None, batch_size=None,
                 alpha_moment=0.9,alpha_variance=0.999,epsilon=1e-8,
                 grad_scaling=False):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    global moments, variances

    if not (moments and variances):
        for layer in model:
            moments.append([zeros(weight.size()) for weight in layer._asdict().values()])
            variances.append([zeros(weight.size()) for weight in layer._asdict().values()])

    with no_grad():

        for _, layer in enumerate(model):
            for __, weight in enumerate(getattr(layer,field) for field in layer._fields):
                if weight.requires_grad:

                    lr_ = lr

                    weight.grad /= batch_size

                    #print(list(layer._fields)[__],weight.grad)

                    if moments:
                        moments[_][__] = alpha_moment * moments[_][__] + (1-alpha_moment) * weight.grad
                        moment_hat = moments[_][__] / (1-alpha_moment**(epoch_nr+1))
                    if variances:
                        variances[_][__] = alpha_variance * variances[_][__] + (1-alpha_variance) * weight.grad**2
                        variance_hat = variances[_][__] / (1-alpha_variance**(epoch_nr+1))
                    if grad_scaling:
                        lr_ *= norm(weight)/norm(weight.grad)

                    weight -= lr_ * (moment_hat if moments else weight.grad) / ((sqrt(variance_hat)+epsilon) if variances else 1)

                    weight.grad = None


##


def empty_states(model, batch_size=1):
    states = []
    for layer in model:
        if type(layer) != FF1 and type(layer) != FF2:
            state = zeros(batch_size, getattr(layer,layer._fields[0]).size(1))
            # if type(layer) == LSTM: # only for regular prop (prop2 is better.)
            #     state = cat([state]*2,dim=1)
            states.append(state)
    return states


##


def load_model(path=None, fresh_meta=None, py_serialize=True):
    if not path: path = config.model_load_path
    if not fresh_meta: fresh_meta = config.fresh_meta
    obj = pickle_load(path+'.pk')
    if obj:
        model, meta = obj
        if py_serialize:
            model = [type(layer)(*[tensor(getattr(layer,field),requires_grad=True) for field in layer._fields]) for layer in model]
        global moments, variances
        if fresh_meta:
            moments, variances = [], []
        else:
            moments, variances = meta
            if py_serialize:
                moments = [[tensor(e) for e in ee] for ee in moments]
                variances = [[tensor(e) for e in ee] for ee in variances]
        return model

def save_model(model, path=None, py_serialize=True):
    if not path: path = config.model_save_path
    if py_serialize:
        model = [type(layer)(*[getattr(layer,field).detach().numpy() for field in layer._fields]) for layer in model]
        meta = [[[e.detach().numpy() for e in ee] for ee in moments],[[e.detach().numpy() for e in ee] for ee in variances]]
    else:
        meta = [moments,variances]
    pickle_save([model,meta],path+'.pk')


def describe_model(model):
    return f'{config.in_size} ' + ' '.join(str(type(layer)) + " " + str(getattr(layer, layer._fields[0]).size(1)) for layer in model)


def combine_models(model1, model2, model1_nograd=True):
    if model1_nograd:
        for layer in model1:
            for k,v in layer._asdict().items():
                v.requires_grad = False
    return model1 + model2


##


def collect_grads(model):
    grads = [zeros(param.size()) for layer in model for param in layer._asdict().values()]
    ctr = -1
    for layer in model:
        for field in layer._fields:
            ctr += 1
            param = getattr(layer,field)
            if param.requires_grad:
                grads[ctr] += param.grad
                param.grad = None

    return grads

def give_grads(model, grads):
    ctr = -1
    for layer in model:
        for field in layer._fields:
            ctr += 1
            param = getattr(layer,field)
            if param.grad:
                param.grad += grads[ctr]
            else: param.grad = grads[ctr]


##


from torch.nn import Module, Parameter

class Convert2TorchModel(Module):

    def __init__(self, model):
        super(Convert2TorchModel, self).__init__()
        for i,layer in enumerate(model):
            converted = [Parameter(getattr(layer,field)) for field in layer._fields]
            for field, value in zip(layer._fields, converted):
                setattr(self,f'layer{i}_{field}',value)
            setattr(self,f'type{i}',type(layer))
            model[i] = (getattr(self, f'type{layer}'))(converted)

    def forward(self, states, inp):
        model = [(getattr(self,f'type{layer}'))(getattr(self,param) for param in dir(self) if f'layer{layer}' in param)
            for layer in range(len(states))]
        prop_model(model, states, inp)
