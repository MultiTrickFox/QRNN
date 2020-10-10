import config

from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.extensions.standard.barrier import Barrier
from qiskit.aqua.components.initial_states import Custom

from torch import Tensor
from numpy import ndarray

from numpy import array
from numpy import pi, sqrt
from numpy import power, abs

from copy import deepcopy


##


def prop_circuit(params, timestep, mode='theoretical', backend=None, hm_trials=1000, show_details=False):

    if type(params)==Tensor: params = params.detach().numpy()
    if type(params)==ndarray: params = params.flatten()
    if type(timestep)==Tensor: timestep = timestep.detach().numpy()
    if type(timestep)==ndarray: timestep = timestep.flatten()

    c = make_circuit(params,timestep)

    if mode == 'theoretical':
        return [abs(e)**2 for e in run_circuit(c,backend='state_vector')]

    elif mode == 'experimental':
        return array([v for k,v in run_circuit(c,backend=backend,hm_trials=hm_trials,display_job_status=show_details).items()])


##


def phase_encoder(c, timestep):

    hm_qbits = len(c.qubits)

    reversed_qbit_ids = list(reversed(range(hm_qbits)))

    distance_checks = [2**i for i in reversed_qbit_ids]

    c.h(reversed_qbit_ids)

    for i,e in enumerate(timestep):

        if e != 0:

            c_e = QuantumCircuit(hm_qbits,hm_qbits)
            xs_to = []

            for j,dist in enumerate(distance_checks):

                if i < dist:
                    c_e.x(reversed_qbit_ids[j])
                    xs_to.append(reversed_qbit_ids[j])
                else:
                    i -=dist

            mcz(c_e,reversed_qbit_ids)

            for x_to in reversed(xs_to):
                c_e.x(x_to)

            c += c_e

    c.barrier()


def amplitude_modifier(hm_qbits=None):

    if not hm_qbits: hm_qbits = config.hm_qbits

    the_circuit = []

    prev_circuit = None

    for current_qbit in reversed(range(hm_qbits)):

        if current_qbit == hm_qbits-1:

            circuit = [['RY', current_qbit]]

            prev_circuit = deepcopy(circuit)
            the_circuit.extend(circuit)

        else:

            circuit = []

            circuit.append(['CX', hm_qbits-1, current_qbit])

            for i, element in enumerate(prev_circuit):
                prev_circuit[i] = [e if ii == 0 else e - 1 for ii, e in enumerate(element)]
            circuit += reversed(prev_circuit)

            half_circuit = deepcopy(circuit)

            circuit.append(['CX', hm_qbits-1, current_qbit])

            circuit += list(reversed(half_circuit))[:-1]

            prev_circuit = deepcopy(circuit)
            the_circuit.extend(circuit)

    return the_circuit


##


def make_circuit(params, timestep, hm_layers=config.circuit_layers):

    params = params * 2*pi

    c = QuantumCircuit(config.hm_qbits, config.hm_qbits)

    if config.reconstruct_qstate:
        c += Custom(config.hm_qbits,state_vector=[e for e in sqrt(timestep)]+[0]*(config.statevec_size-config.timestep_size)).construct_circuit()
        #dis very finally. #c.initialize([e for e in sqrt(timestep)]+[0]*(config.statevec_size-config.timestep_size), range(config.hm_qbits))

    ctr = 0

    for i in range(hm_layers):

        if config.ansatz_mode == 0:

            the_circuit = amplitude_modifier()

            for element in the_circuit:

                if element[0] == 'RY':

                    c.ry(params[ctr],element[1])
                    ctr +=1

                elif element[0] == 'CX':

                    c.cx(element[1],element[2])

        elif config.ansatz_mode == 1:

            for id in range(config.hm_qbits):
                c.u3(params[ctr], params[ctr+1], params[ctr+2], id)
                ctr +=3

        make_entangle(c)

    c.barrier()

    return c


def make_entangle(c):

    if config.entangle_mode == -1:

        pass

    elif config.entangle_mode == 0:

        for id in range(config.hm_qbits-1):
            c.cx(id, id+1)
        c.cx(config.hm_qbits-1, 0)

    elif config.entangle_mode == 1:

        wire_ids = list(range(config.hm_qbits))
        for id_from in wire_ids:
            for id_to in wire_ids:
                if id_from != id_to:
                    c.cx(id_from, id_to)


##


def swap_bit_rank(c):
    hm_qbits = c.num_qubits
    max_index = hm_qbits-1
    for i in range(hm_qbits//2):
        c.swap(i,max_index-i)


def mcz(c, *args):
    c.h(args[-1])
    c.mcx(args[:-1],args[-1])
    c.h(args[-1])


##


def find_backend(show_details=False, avoid_16=True):

    min_qbits = config.hm_qbits
    min_pending = 999_999
    min_backend = None

    try:
        IBMQ.load_account()
        provider = IBMQ.get_provider("ibm-q")
    except Exception as e:
        print(f'Error: could not get IBMQ provider, {e}')
        assert None

    if config.preferred_backend:
        return config.preferred_backend

    for backend in provider.backends():

        hm_jobs = backend.status().pending_jobs

        try:
            qbit_count = len(backend.properties().qubits)

            if hm_jobs <= min_pending and qbit_count >= min_qbits:
                if avoid_16 and qbit_count != 16:
                    min_pending = hm_jobs
                    min_backend = backend

        except: pass

    try:
        assert min_backend
        if show_details: print(f'picked backend: {min_backend.name()}')
        return min_backend

    except:
        print(f'Error: Backend supporting {min_qbits} qbits not found, defaulting to QASM \n')
        return provider.get_backend('ibmq_qasm_simulator')


def run_circuit(circuit,
                measure   = True,
                hm_trials = 1_000,
                backend   = None,
                reverse=False, normalize=True, fill_zero_counts=True, order=True,
                draw=False, draw_optimized=False, display_job_status=False,
                ):

    if not backend:
        backend = Aer.get_backend('qasm_simulator')

    elif type(backend) == str:

        if 'state' in backend:
            backend = Aer.get_backend('statevector_simulator')
            state_vector = execute(circuit, backend).result().get_statevector()
            return state_vector

        else:
            try:
                from warnings import filterwarnings
                filterwarnings("ignore")
                IBMQ.load_account()
                provider = IBMQ.get_provider("ibm-q")
                backend = provider.get_backend(backend)
            except Exception as e:
                print(f'Error: could not get IBMQ provider, {e}')
                assert None

    if display_job_status:
        print(f'running on: {backend.name()}')

    if measure:
        circuit.measure(range(len(circuit.clbits)), range(len(circuit.clbits)))

    if draw:
        print(circuit.draw())

    circuit.data = [e for e in circuit.data if type(e[0]) != Barrier]
    if draw_optimized:
        print(circuit.draw())

    job = execute(circuit, backend=backend, shots=hm_trials)

    if display_job_status:
        job_monitor(job)

    try:
        result = job.result()
        counts = result.get_counts(circuit)

        if fill_zero_counts:
            hm_qbits = len(list(counts.keys())[0])
            for nr in range(2**hm_qbits):
                s = str(bin(nr))[2:]
                if len(s) != hm_qbits:
                    for _ in range(hm_qbits-len(s)):
                        s = "0"+s
                if s not in counts.keys():
                    counts[s] = 0.0
        if reverse:
            counts = {k[-1::-1]:v for k, v in counts.items()}
        if normalize:
            counts = {k:v/hm_trials for k, v in counts.items()}
        if order:
            counts = dict(sorted(counts.items()))

        return counts

    except Exception as e: print(f'Error: job failed with {job.error_message()}, {e}')


##
