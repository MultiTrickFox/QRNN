import config

from ext import cls
from data import note_dict, note_reverse_dict, empty_vector, normalize_vector
from data import convert_to_stream
from model import load_model, empty_states, prop_model_nocircuit
from circuit import find_backend, prop_circuit
from simu import prop_circuits

from torch import tensor, Tensor, zeros
from torch import no_grad, float32

from random import choices


##


states, outs, track = [], [], []


def main():

    if not (model := load_model()):
        input(f'Error: model not found, run training..')
        return

    global states, outs, track

    if not states:
        states = [empty_states(model)]
    else:
        possible_states = empty_states(model)
        inherited_states = states[-1]
        if len(possible_states) != len(inherited_states):
            states = [empty_states(model)]
        elif not all(layer_state_1.size(1)==layer_state_2.size(1) for layer_state_1,layer_state_2 in zip(possible_states,inherited_states)):
            states = [empty_states(model)]

    while 1:

        cls()
        if (inp := input('Select: \n\t(B)uild \n\t(S)how \n\t(P)lay \n\t(R)emoveLast \n\t(C)lear \n\n      > ')) == '': break
        cls()

        if inp.lower()[0] == 'u':

            toggle_online = False

            time_signature = 4

            if not outs:

                if (inp:=input('Enter a note: ').upper()) != '':

                    if inp[-1] == ';':
                        toggle_online = True
                        inp = inp[:-1]

                    found = False
                    while not found:
                        found = all([e in note_dict.keys() for e in inp.split(',')])
                        if not found:
                            inp = input('> example: G,B,F# or g,b \nEnter a note: ').upper()
                        else: break

                    track.append(inp)
                    outs.append(human_2_ai(inp))

            if outs:

                try: hm_timesteps = (int(input('amount of bars: ')))*config.beat_resolution*time_signature
                except:
                    hm_timesteps = config.hm_bars_grouped*config.beat_resolution*time_signature
                    print(f'> set to {config.hm_bars_grouped}')

                if not toggle_online:
                    try:
                        config.note_pick_mode = int(input('note pick mode: '))
                        config.polyphony = (config.note_pick_mode != 3)
                    except: print(f'> set to {config.note_pick_mode}')
                else:
                    config.polyphony = False

                print(f'unrolling {hm_timesteps} steps..')

                for _ in range(hm_timesteps):

                    if (_:=_+1)%50 == 0:
                        print(_)

                    out, state, [theo,exp,final_exp] = \
                        prop_model(model, states[-1], outs[-1], online_collapse=toggle_online)

                    states.append(state)

                    if toggle_online:
                        resp = ai_2_human(final_exp)

                    else:

                        if config.note_pick_mode == 0:
                            theoretical = [[note_reverse_dict[id], prob] for id, prob in enumerate(theo) if id < config.timestep_size]
                            theoretical = sorted(theoretical, key=lambda x: x[1], reverse=True)
                            resp = theoretical[0][0]

                        elif config.note_pick_mode == 1:
                            experimental = [[note_reverse_dict[id], prob] for id, prob in enumerate(exp) if id < config.timestep_size]
                            experimental = sorted(experimental, key=lambda x: x[1], reverse=True)
                            resp = experimental[0][0]

                        else:

                            resp = ai_2_human(out)

                    track.append(resp)

                    out_ = zeros(1,config.timestep_size)
                    out_ += out[:,:config.timestep_size]
                    for i in range(config.timestep_size, config.statevec_size):
                        out_[-1] += out[:,i]
                    outs.append(out_)

        elif inp.lower()[0] == 'b':

            enter_manually = True
            toggle_online = False

            while 1:

                if enter_manually:

                    if (inp := str(input('> Enter a note: ')).upper()) == '':
                        break

                    if inp[-1] == ';':
                        toggle_online = True
                        inp = inp[:-1]

                    if all([e in note_dict.keys() for e in inp.split(',')]):
                        track.append(inp)
                        outs.append(human_2_ai(inp))
                        enter_manually = False

                else:

                    out, state, [theo,exp,final_exp] = \
                        prop_model(model, states[-1], outs[-1], online_collapse=toggle_online)

                    states.append(state)

                    theoretical = [[note_reverse_dict[id],prob] for id,prob in enumerate(theo) if id<config.timestep_size]
                    theoretical = sorted(theoretical, key=lambda x: x[1], reverse=True)

                    experimental = [[note_reverse_dict[id],prob] for id,prob in enumerate(exp) if id<config.timestep_size]
                    experimental = sorted(experimental, key=lambda x: x[1], reverse=True)

                    print(f'\n> Theoretical q-state: {theoretical}\n')
                    print(f'> Experimental q-state: {experimental}\n')
                    if toggle_online:
                        print(f'\n> Real Experimental q-state: {final_exp}\n')

                    resp = ai_2_human(final_exp)

                    if input(f'> Collapsed to: {resp} - Keep it? (y/n): ').lower() == 'y':
                        track.append(resp)
                        out_ = zeros(1, config.timestep_size)
                        out_ += out[:,:config.timestep_size]
                        for i in range(config.timestep_size, config.statevec_size):
                            out_[-1] += out[:,i]
                        outs.append(out_)
                        enter_manually = False
                        print('collapse added')
                    else:
                        enter_manually = True
                        print('collapse not added')

                    if toggle_online:
                        toggle_online = False

        elif inp.lower()[0] == 's' and track:
            convert_to_stream(track).plot(title='my musiplot')

        elif inp.lower()[0] == 'p' and track:
            try:
                convert_to_stream(track).show()
            except Exception: pass

        elif inp.lower()[0] == 'r' and track:
            del track[-1]
            del states[-1]
            del outs[-1]

        elif inp.lower()[0] == 'c':
            outs, track, states = [], [], [empty_states(model)]

        else:
            pass


##


def human_2_ai(note_names):

    vector = empty_vector.copy()

    for note_name in note_names.split(','):
        note_id = note_dict[note_name.upper()]
        vector[note_id] +=1

    vector = normalize_vector(vector)

    return tensor(vector, dtype=float32).view(1,len(vector))


def prop_model(model, states, inp, online_collapse=False):

    with no_grad():

        out, new_states = prop_model_nocircuit(model,states,inp)

        if config.act_classical_rnn:

            print(f'output: {out}')

            theoretical, experimental, real_experimental = [],[],[]

        else:

            theoretical = prop_circuit(out,inp,'theoretical')
            experimental = prop_circuit(out,inp,'experimental')

            real_experimental = experimental
            if online_collapse:
                real_experimental = prop_circuit(out,inp,'experimental',find_backend(),show_details=True,hm_trials=1 if not config.polyphony else 1_000)

            out = prop_circuits(out, inp)

        return out, new_states, [theoretical,experimental,real_experimental]


def ai_2_human(out):

    out = out.flatten()

    if not config.polyphony:
        collapsed = choices(range(len(out)),weights=out,k=1)[0]
        for i in range(len(out)):
            out[i] = 1. if i==collapsed else .0

    out_converted = ""

    for ii, i in enumerate(out):

        if i >= 1/config.statevec_size:

            if element := note_reverse_dict.get(ii):
                out_converted += element
            else:
                out_converted += "R"

            out_converted += ","

    out_converted = out_converted[:-1]

    return out_converted



##



if __name__ == '__main__':
    main()