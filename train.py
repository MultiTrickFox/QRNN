import config

from ext import parallel, now
from model import respond_to, sequence_loss, sgd, adaptive_sgd
from model import make_model, save_model, load_model, describe_model, empty_states
from model import collect_grads, give_grads
from model import prop_model_nocircuit
from simu import prop_circuits
from data import preprocess, save_data, load_data, split_dataset, batchify

from torch import zeros
from torch import cat, stack
from torch import softmax
from torch import no_grad

from matplotlib.pyplot import plot, show

from copy import deepcopy

from math import ceil

from time import time


##


def main(model=None):

    print(f'readying model & data @ {now()}')

    data = load_data()
    if not data:
        save_data(preprocess())
        data = load_data()

    if not model:
        if not config.fresh_model:
            model = load_model()
        if not model:
            model = make_model()
            save_model(model)
            model = load_model()
            print('created ',end='')
        else: print('loaded ',end='')
        print(f'model: {describe_model(model)}')

    print(f'total files: {len(data)}, ',end='')

    data, data_dev = split_dataset(data)

    if config.batch_size > len(data):
        config.batch_size = len(data)
    elif config.batch_size == -1:
        config.batch_size = len(data_dev)

    print(f'train: {len(data)}, dev: {len(data_dev)}, batch size: {config.batch_size}')

    print(f'hm train: {sum(len(datapoint) for datapoint in data)}, '
          f'hm dev: {sum(len(datapoint) for datapoint in data_dev)}, '
          f'learning rate: {config.learning_rate}, '
          f'optimizer: {config.optimizer}, '
          f'\ntraining for {config.hm_epochs} epochs.. ',end='\n')

    one_batch = (config.batch_size == len(data)) or (config.train_combined and config.train_parallel)
    config.shuffle_epoch &= not one_batch
    window_slide_multiplier = config.hm_bars_grouped//config.hm_bars_slide
    if config.ckp_save_epochs == -1: config.ckp_save_epochs = range(config.hm_epochs)

    data_losss, dev_losss = [], []

    if config.initialize_loss:

        print(f'initializing losses @ {now()}', flush=True)
        if not one_batch:
            data_losss.append(dev_loss(model,data))
        dev_losss.append(dev_loss(model,data_dev))
        print(f'initial losses: {data_losss, dev_losss}')

    print(f'training started @ {now()}', flush=True)

    for ep in range(config.hm_epochs):

        loss = 0

        if config.train_parallel and config.train_combined:
            l, g = process_data_onebatch(model, data)
            loss += l
            give_grads(model, g)
            batch_size = sum(sum(len(inp) * window_slide_multiplier for inp, lbl in datapoint) for datapoint in data)
            sgd(model, batch_size=batch_size) if config.optimizer == 'sgd' else adaptive_sgd(model, ep, batch_size=batch_size)

        else:
            for i,batch in enumerate(batchify(data)):

                if config.disp_batches:
                    print(f'\tbatch {i}, {sum(len(datapoint) for datapoint in batch)}', end='', flush=True)

                batch_size = sum(sum(len(inp)*window_slide_multiplier for inp,lbl in datapoint) for datapoint in batch)

                if config.train_parallel:
                    l,g = process_batch_parallel(model,batch)
                    loss += l
                    give_grads(model,g)

                elif config.train_combined:
                    loss += process_batch_combined(model, batch)

                else:
                    for j,datapoint in enumerate(batch):
                        states = None
                        for k,(inp,lbl) in enumerate(datapoint):
                            out, states = respond_to(model, inp, states)
                            states = [state.detach() for state in states]
                            loss += sequence_loss(lbl,out)

                sgd(model,batch_size=batch_size) if config.optimizer == 'sgd' else adaptive_sgd(model,ep,batch_size=batch_size)

                if config.disp_batches:
                    print(f', completed @ {now()}' ,flush=True)

        loss /= sum(sum(len(inp)*window_slide_multiplier for inp,lbl in datapoint) for datapoint in data)

        data_losss.append(loss)
        dev_losss.append(dev_loss(model,data_dev))
        
        print(f'epoch {ep}, loss {loss}, dev loss {dev_losss[-1]}, completed @ {now()}', flush=True)

        if ep in config.ckp_save_epochs:
            save_model(model,f'{config.model_save_path}_ckp{ep}')

    data_losss.append(dev_loss(model,data))
    dev_losss.append(dev_loss(model,data_dev))

    print(f'final losses: {[data_losss[-1],dev_losss[-1]]}')

    print(f'training ended @ {now()}', flush=True)

    plot(data_losss)
    show()
    plot(dev_losss)
    show()

    if config.overwrite_model or input(f'Save model as {config.model_save_path}? (y/n): ').lower() == 'y':
        save_model(load_model(),config.model_save_path+'_prev')
        save_model(model)

    return model, [data_losss, dev_losss]


##


def grad_loss(args):

    model, datapoint = args

    states = None
    loss = 0

    grads = [zeros(param.size()) for layer in model for param in layer._asdict().values()]

    for inp,lbl in datapoint:

        out, states = respond_to(model, inp, states)

        states = [state.detach() for state in states]

        loss += sequence_loss(lbl, out)
        grads = [e1 + e2 for e1, e2 in zip(grads, collect_grads(model))]

    return grads, loss


def nograd_loss(args):

    model, datapoint = args

    states = None
    loss = 0

    with no_grad():

        for inp,lbl in datapoint:

            out, states = respond_to(model, inp, states)

            loss += sequence_loss(lbl, out, do_grad=False)

    return loss


##


def process_batch_parallel(model, batch):

    loss = 0
    grads = [zeros(param.size()) for layer in model for param in layer._asdict().values()]

    for result in parallel(grad_loss, [[model, datapoint] for datapoint in batch]):
        grads = [e0 + e1 for e0, e1 in zip(grads, result[0])]
        loss += result[1]

    return loss, grads


def process_batch_combined(model,batch,training_run=True):

    batch = deepcopy(batch)

    loss = 0

    zero_states = empty_states(model,len(batch))

    all_states = deepcopy(zero_states)

    window_slide_ratio = config.hm_bars_slide/config.hm_bars_grouped
    teacher_ratio = config.hm_bars_teacher/config.hm_bars_grouped

    max_inplbls = max(len(datapoint) for datapoint in batch)

    for datapoint in batch:
        hm = max_inplbls-len(datapoint)
        if hm:
            datapoint.extend([None]*hm)

    has_remaining_inplbl = list(range(len(batch)))

    for ctr_inplbl in range(max_inplbls):

        # print(f'\t',f'{ctr_inplbl}/{max_inplbls}',now(),flush=True)

        has_remaining_inplbl = [i for i in has_remaining_inplbl if batch[i][ctr_inplbl] is not None]

        inplbls_slice = [batch[i][ctr_inplbl] for i in has_remaining_inplbl]

        max_inplen = max(len(inp) for inp,lbl in inplbls_slice)

        for inp,lbl in inplbls_slice:
            hm = max_inplen-len(inp)
            if hm:
                inp.extend([None]*hm)

        all_inps = [batch[i][ctr_inplbl][0] for i in has_remaining_inplbl]
        all_lbls = [batch[i][ctr_inplbl][1] for i in has_remaining_inplbl]

        states_transfers_to = [int((len(inp)+1)*window_slide_ratio) for inp,lbl in inplbls_slice]
        states_to_transfer = deepcopy(zero_states)

        teacher_up_to = [int((len(inp)+1)*teacher_ratio) for inp,lbl in inplbls_slice]

        # all_outs = []

        has_remaining_inp = list(has_remaining_inplbl)
        has_remaining_inp_= range(len(has_remaining_inplbl))

        for t in range(max_inplen):

            has_remaining_inp = [i for i,ii in zip(has_remaining_inp,has_remaining_inp_) if all_inps[ii][t] is not None]

            links_to_prev = [has_remaining_inp_.index(i) for i in [has_remaining_inplbl.index(i) for i in has_remaining_inp]]

            has_remaining_inp_= [has_remaining_inplbl.index(i) for i in has_remaining_inp]

            # inps = cat([all_inps[i][t] for i in has_remaining_inp_], dim=0)
            # lbls = cat([all_lbls[i][t] for i in has_remaining_inp_], dim=0)

            inps = cat([all_inps[i][t] if t <= teacher_up_to[i] else outs[links_to_prev[ii]:links_to_prev[ii]+1,:config.timestep_size] for ii,i in enumerate(has_remaining_inp_)], dim=0)
            lbls = cat([all_lbls[i][t] for i in has_remaining_inp_], dim=0)
            
            states = [stack([row for i,row in enumerate(layer_state) if i in has_remaining_inp]) for layer_state in all_states]

            #start = time()

            outs, states = prop_model_nocircuit(model, states, inps)

            for layer_state, state in zip(all_states, states):
                for ii,i in enumerate(has_remaining_inp):
                    layer_state[i] = state[ii]

            t +=1
            for i in has_remaining_inp_:
                if t == states_transfers_to[i]:
                    for layer_state, transfer_state in zip(all_states, states_to_transfer):
                        transfer_state[i] = layer_state[i].detach()

            #nnt = time() - start

            # TDO : start a thread with this prop circuit + its loss part?

            #start = time()

            if not config.act_classical_rnn:

                outs = prop_circuits(outs, inps)

                outs_ = outs[:,:config.timestep_size]
                for i in range(config.timestep_size,config.statevec_size):
                    outs_[:,-1] += outs[:,i]
                outs = outs_
                
            # else:
            #     outs = softmax(outs,dim=1)
            #outs = outs/outs.sum()

            #cct = time() - start

            # print('circuit out',flush=True) ; show_it = 7
            # print(circ_outs[show_it])
            # print('extra qiskit answer',flush=True)
            # from circuit import make_circuit,run_circuit
            # arg2 = inps[show_it]
            # arg1 = outs[show_it]
            # results = run_circuit(make_circuit(arg1.detach().numpy(),arg2.detach().numpy()),backend='state',display_job_status=False)
            # result_final = list(abs(result)**2 for result in results)
            # print(result_final)
            # input('Halt..')
            # from circuit import prop_circuit
            # print('extra extra answers..')
            # print('theoretical:')
            # print(prop_circuit(arg1,arg2))
            # print('experimental:')
            # print(prop_circuit(arg1,arg2,mode='experimental'))
            # input("HALT!")

            # print(f'> training times for t {t}/{max_inplen}*{max_inplbls}: {nnt} - {cct}  ;; {cct/nnt}')
            # input("continue to next it.. ?")

            # all_outs.append(circ_outs)

            loss += sequence_loss(lbls,outs,do_stack=False)

            # for i,layer in enumerate(model):
            #     for l in layer._fields:
            #         g = getattr(layer,l).grad
            #         if g is not None:
            #             if g.sum() == 0: print(f'Zero grad at layer {i} {l}')
            #             else: print(f'layer {i} {l} norm: {g.norm()}, sum: {g.sum()}, abs-sum: {g.abs().sum()}')
            #         else: print(f'No grad at layer {i} {l} !')
            # input('Halt !')

        all_states = states_to_transfer

    if training_run:
        loss.backward()

    return float(loss)


def wrapper(args):
    model,data = args
    loss = process_batch_combined(model,data)
    return collect_grads(model), loss

def process_data_onebatch(model, data, chunk_size=None):

    if not chunk_size: chunk_size = config.batch_size

    loss = 0
    grads = [zeros(param.size()) for layer in model for param in layer._asdict().values()]

    #data_size = sum(sum(len(inp)*config.hm_bars_grouped//config.hm_bars_slide for inp,lbl in datapoint) for datapoint in data)

    for result in parallel(wrapper, [[model, data[i*chunk_size:(i+1)*chunk_size]] for i in range(ceil(len(data)/config.batch_size))]):
        grads = [e0+e1 for e0,e1 in zip(grads,result[0])]
        loss += result[1]

    #loss /= data_size

    return loss, grads


##


def dev_loss(model,data):

    window_slide_multiplier = config.hm_bars_grouped//config.hm_bars_slide

    loss = 0

    batch_size = sum(sum(len(inp)*window_slide_multiplier for inp, lbl in datapoint) for datapoint in data)

    if config.train_parallel and config.train_combined:
        l, _ = process_data_onebatch(model, data)
        return l /batch_size

    for i in range(ceil(len(data)/config.batch_size)):

        batch = data[i*config.batch_size:(i+1)*config.batch_size]

        # if config.disp_batches:
        #     print(f'\tbatch {i}, {sum(len(datapoint) for datapoint in batch)}', end='', flush=True)

        if config.train_parallel:
            loss += sum(result for result in parallel(nograd_loss,[[model,datapoint] for datapoint in batch]))
        elif config.train_combined:
            with no_grad():
                loss += process_batch_combined(model, batch, training_run=False)
            # collect_grads(model)
        else:
            loss += sum(nograd_loss([model,datapoint]) for datapoint in batch)

        # if config.disp_batches:
        #     print(f', completed @ {now()}', flush=True)

    return loss /batch_size

    # loss, grads = process_data_onebatch(model, data)
    # return loss


## extra(s)


def eval_model_score(model,data):

    from torch import ones
    from numpy import corrcoef, log
    from numpy import sum as nsum
    from numpy.linalg import eig

    data = deepcopy(data)
    model = deepcopy(model)
    for layer in model:
        for field in layer._fields:
            getattr(layer,field).requires_grad = False

    for i in range(ceil(len(data)/config.batch_size)):
        batch = data[i*config.batch_size:(i+1)*config.batch_size]
        # print(f'\tbatch {i}, {sum(len(datapoint) for datapoint in batch)}, started @ {now()}', flush=True)
        for file in batch:
            for inplbl in file:
                for timestep in inplbl[0]:
                    timestep.requires_grad = True
        zero_states = empty_states(model, len(batch))
        all_states = deepcopy(zero_states)
        window_slide_ratio = config.hm_bars_slide/config.hm_bars_grouped
        teacher_ratio = config.hm_bars_teacher/config.hm_bars_grouped
        max_inplbls = max(len(datapoint) for datapoint in batch)
        for datapoint in batch:
            hm = max_inplbls - len(datapoint)
            if hm:
                datapoint.extend([None]*hm)
        has_remaining_inplbl = list(range(len(batch)))

        for ctr_inplbl in range(max_inplbls):
            has_remaining_inplbl = [i for i in has_remaining_inplbl if batch[i][ctr_inplbl] is not None]
            inplbls_slice = [batch[i][ctr_inplbl] for i in has_remaining_inplbl]
            max_inplen = max(len(inp) for inp, lbl in inplbls_slice)
            for inp, lbl in inplbls_slice:
                hm = max_inplen - len(inp)
                if hm:
                    inp.extend([None]*hm)
            all_inps = [batch[i][ctr_inplbl][0] for i in has_remaining_inplbl]
            states_transfers_to = [int((len(inp)+1)*window_slide_ratio) for inp, lbl in inplbls_slice]
            states_to_transfer = deepcopy(zero_states)
            teacher_up_to = [int((len(inp)+1)*teacher_ratio) for inp, lbl in inplbls_slice]
            has_remaining_inp = list(has_remaining_inplbl)
            has_remaining_inp_ = range(len(has_remaining_inplbl))

            for t in range(max_inplen):
                has_remaining_inp = [i for i, ii in zip(has_remaining_inp, has_remaining_inp_) if all_inps[ii][t] is not None]
                links_to_prev = [has_remaining_inp_.index(i) for i in [has_remaining_inplbl.index(ii) for ii in has_remaining_inp]]
                has_remaining_inp_ = [has_remaining_inplbl.index(i) for i in has_remaining_inp]
                inps = cat([all_inps[i][t] if t < teacher_up_to[i] else outs[links_to_prev[ii]:links_to_prev[ii]+1,:config.timestep_size] for ii, i in enumerate(has_remaining_inp_)], dim=0)
                states = [stack([row for i, row in enumerate(layer_state) if i in has_remaining_inp]) for layer_state in all_states]
                outs, states = prop_model_nocircuit(model, states, inps)
                for layer_state, state in zip(all_states, states):
                    for ii, i in enumerate(has_remaining_inp):
                        layer_state[i] = state[ii]
                t += 2
                for i in has_remaining_inp_:
                    if t == states_transfers_to[i]:
                        for layer_state, transfer_state in zip(all_states, states_to_transfer):
                            transfer_state[i] = layer_state[i].detach()
                if not config.act_classical_rnn:
                    outs = prop_circuits(outs, inps)
                # else:
                #     outs = softmax(outs,dim=1)
                outs.backward(ones(outs.size()),retain_graph=True)

            all_states = states_to_transfer

    data_grad = []
    for file in data:
        file_grad = zeros(1,config.timestep_size)
        for inplbl in file:
            if inplbl is not None:
                inp_seq = inplbl[0]
                # inp_seq_grad = zeros(1,config.timestep_size)
                for timestep in inp_seq:
                    if timestep is not None and timestep.grad is not None:
                        file_grad += timestep.grad.detach()
                #inp_seq_grad /= len(inp_seq)*window_slide_multiplier
                #file_grad.append(inp_seq_grad)
        #file_grad = cat(file_grad,dim=1)
        window_slide_multiplier = config.hm_bars_grouped//config.hm_bars_slide
        file_grad /= sum(len(inplbl[0])*window_slide_multiplier for inplbl in file if inplbl)
        data_grad.append(file_grad)
    data_grad = cat(data_grad,dim=0)
    correlations = corrcoef(data_grad)
    v,_ = eig(correlations)
    score = -nsum(log(v+1e-5)+1./(v+1e-5))
    return score, correlations


##



if __name__ == '__main__':
    main()
