import config

from ext import pickle_save, pickle_load

from glob import glob

from pretty_midi import PrettyMIDI

from music21 import *

from torch import tensor, float32

from random import shuffle

from math import ceil

from copy import deepcopy


##


note_dict = {
    'A' : 0,
    'A#': 1, 'B-': 1,
    'B' : 2,
    'C' : 3,
    'C#': 4, 'D-': 4,
    'D' : 5,
    'D#': 6, 'E-': 6,
    'E' : 7,
    'F' : 8,
    'F#': 9, 'G-': 9,
    'G' :10,
    'G#':11, 'A-': 11,
    'R' :12
}

note_reverse_dict = {
    0: 'A',
    1: 'A#',
    2: 'B',
    3: 'C',
    4: 'C#',
    5: 'D',
    6: 'D#',
    7: 'E',
    8: 'F',
    9: 'F#',
    10:'G',
    11:'G#',
    12:'R'
}

empty_vector = [0 for _ in range(config.in_size)]


##


def preprocess():

    data = []

    raw_files = glob(config.samples_folder + "/**/*.mid*") + glob(config.samples_folder + "/**/*.MID*") + glob(config.samples_folder + "/*.mid*") + glob(config.samples_folder + "/*.MID*")

    print(f'files to read: {len(raw_files)}')

    for i,raw_file in enumerate(raw_files):

        try:
            data.extend(preprocess_file(raw_file))
        except Exception as e: print(f'ERROR: {raw_file} failed, {e}')

        if (i+1)%50==0:
            print(f'>> {i+1}/{len(raw_files)}')

    print(f'>> obtained total of {len(data)} sequences.')

    return data


def preprocess_file(raw_file):

    print(f'> processing file {raw_file}')

    ## remove drums

    sound = PrettyMIDI(raw_file)
    drum_instruments_index = [i for i,inst in enumerate(sound.instruments) if inst.is_drum]
    for i in sorted(drum_instruments_index, reverse=True):
        del sound.instruments[i]
    sound.write(raw_file)

    ## read parts

    sample = converter.parse(raw_file)

    parts = instrument.partitionByInstrument(sample)
    if not parts:
        parts = [sample.flat]

    try:
        time_signatures = [int(part.timeSignature.ratioString[0]) for part in parts]
    except:
        print('WARNING: time signature failed, check the file')
        time_signatures = [4 for _ in range(len(parts))]

    ## convert parts

    converted_sequences = []

    for part, time_signature in zip(parts, time_signatures):

        converted_sequence = [[] for _ in range(len(part.makeMeasures())*time_signature*config.beat_resolution)]

        for element in part.flat:

            try:
                assert element.beat
                assert element.duration

                add_to_sequence(converted_sequence, element)

            except:
                pass

        converted_sequences.append(converted_sequence)

    ## combine parts

    if config.combine_instrus:

        combined_converted_sequence = []

        max_len = max([len(part) for part in converted_sequences])

        for part in converted_sequences:
            if len(part) != max_len:
                for _ in range(max_len-len(part)):
                    part.append([])

        for t in range(max_len):
            t_collection = []
            for part in converted_sequences:
                t_collection.extend(part[t])
            combined_converted_sequence.append(t_collection)

        converted_sequences = [combined_converted_sequence]

    ## finalize

    converted_sequences = [[normalize_vector(vectorize_timestep(timestep)) for timestep in trim_empty_timesteps(converted_sequence,time_signature)]
                            for converted_sequence,time_signature in zip(converted_sequences,time_signatures)]

    # for converted_sequence, time_signature in zip(converted_sequences, time_signatures):
    #     # if input('Show stream? (y/n): ').lower() == 'y':
    #     # from numpy.random import rand
    #     # if rand() <= .5:
    #         from interact import convert_to_stream
    #         sequence_string = [''.join(f'{note_reverse_dict[i]},' for i, element in enumerate(timestep) if element > 0)[:-1] for timestep in converted_sequence]
    #         convert_to_stream(sequence_string).show()
    #         input('Halt..')

    return zip(converted_sequences,time_signatures)


##


def add_to_sequence(converted_sequence, element):

    if isinstance(element, note.Note):
        vector = vectorize_element(element)
        starting_group = round(element.offset*config.beat_resolution)
        ending_group = int(element.duration.quarterLength*config.beat_resolution)
        if ending_group == 0: ending_group = 1
        for group in range(ending_group):
            converted_sequence[starting_group+group].append(vector)

    elif isinstance(element, chord.Chord):
        starting_group = round(element.offset*config.beat_resolution) # normally this should've acted on "for each e in chord", thank you music21..
        ending_group = int(element.duration.quarterLength*config.beat_resolution)
        if ending_group == 0: ending_group = 1
        for e in element:
            vector = vectorize_element(e)
            for group in range(ending_group):
                converted_sequence[starting_group+group].append(vector)


##


def vectorize_element(element):

    vector = empty_vector.copy()
    vector[note_dict[element.pitch.name]] += 1

    return vector


def vectorize_timestep(timestep):

    vec = empty_vector.copy()

    if config.polyphony:

        for vector in timestep:
            for i,v in enumerate(vector):
                vec[i] +=v

    else:

        counts = {}

        for vector in timestep:
            counts[str(vector)] = 0.0
        for vector in timestep:
            counts[str(vector)] += 1

        if counts:
            max_val = max(list(counts.values()))
            for k,v in counts:
                if v == max_val:
                    vec[k] += 1

    if vec == empty_vector:
        vec[-1] +=1

    return vec


def normalize_vector(vector):

    return [e/sum(vector) for e in vector]


##


def trim_empty_timesteps(converted_sequence, time_signature):

    trim_groups_of = time_signature*config.beat_resolution

    trim_from_start = 0
    for i in range(int(len(converted_sequence)/trim_groups_of)):
        if not any(converted_sequence[i*trim_groups_of+j] for j in range(trim_groups_of)):
            trim_from_start += 1
        else:
            break
    if trim_from_start:
        converted_sequence = converted_sequence[trim_from_start*trim_groups_of:]

    trim_from_end = 0
    for i in range(int(len(converted_sequence)/trim_groups_of)-1, -1, -1):
        if not any(converted_sequence[i*trim_groups_of+j] for j in range(trim_groups_of)):
            trim_from_end += 1
        else:
            break
    if trim_from_end:
        converted_sequence = converted_sequence[:-trim_from_end*trim_groups_of]

    trim_from_mid = []
    for i in range(int(len(converted_sequence)/trim_groups_of)):
        if not any(converted_sequence[i*trim_groups_of+j] for j in range(trim_groups_of)):
            trim_from_mid.append(i)
    if trim_from_mid:
        converted_sequence_ = []
        for i in range(int(len(converted_sequence)/trim_groups_of)):
            if i not in trim_from_mid:
                converted_sequence_.extend(converted_sequence[i*trim_groups_of:(i+1)*trim_groups_of])
        converted_sequence = converted_sequence_

    return converted_sequence


##


def save_data(data, path=None):
    if not path: path = config.data_path
    pickle_save(data, path+'.pk')


def load_data(path=None):
    if not path: path = config.data_path
    data = pickle_load(path+'.pk')

    if data:

        for i,(sequence,time_sig) in enumerate(data):

            file_data = []

            window_size = int(config.hm_bars_grouped*time_sig*config.beat_resolution)
            window_slide = int(config.hm_bars_slide*time_sig*config.beat_resolution)

            hm_windows = ceil(len(sequence)/window_slide)

            for sub_sequence in [sequence[i*window_slide:i*window_slide+window_size] for i in range(hm_windows)]:
                
                inp = [tensor(timestep,dtype=float32).view(1,config.in_size) for timestep in sub_sequence[:-1]]
                lbl = [tensor(timestep,dtype=float32).view(1,config.in_size) for timestep in sub_sequence[1:]]

                # if not config.act_classical_rnn:
                #     lbl = [tensor(timestep+[0]*(config.statevec_size-config.in_size),dtype=float32).view(1,config.statevec_size) for timestep in sub_sequence[1:]]
                # else:
                #     lbl = [tensor(timestep, dtype=float32).view(1, config.in_size) for timestep in sub_sequence[1:]]

                file_data.append([inp,lbl])

            data[i] = file_data

        return data


def split_dataset(data, dev_ratio=config.dev_ratio, do_shuffle=config.shuffle_split):
    if do_shuffle: shuffle(data)
    hm_train = ceil(len(data)*(1-dev_ratio))
    data_dev = data[hm_train:]
    data = data[:hm_train]
    return data, data_dev


def batchify(data, batch_size=config.batch_size, do_shuffle=config.shuffle_epoch):
    shuffle(data) if do_shuffle else None
    hm_batches = int(len(data)/batch_size)
    return [data[i*batch_size:(i+1)*batch_size] for i in range(hm_batches)] \
        if hm_batches else [data]


##


def convert_to_stream(track):

    track = [timestep.split(',') for timestep in track]

    music_stream = stream.Stream()
    music_stream.timeSignature = meter.TimeSignature(f'4/4')
    music_stream.insert(0, metadata.Metadata(
        title='vanilla ai',
        composer=f'sent from {config.model_load_path}'))

    for i,timestep in enumerate(track):

        c = chord.Chord()

        for note_name in timestep:

            sustain = 1

            for other_timestep in track[i+1:]:

                sustains_to_other = False

                for ii,other_note_name in enumerate(other_timestep):

                    if note_name == other_note_name:
                        sustains_to_other = True
                        sustain +=1
                        del other_timestep[ii]

                        if 'R' in other_timestep:
                            del other_timestep[other_timestep.index('R')]

                        break

                if not sustains_to_other:

                    break

            if note_name != 'R':
                n = note.Note(note_name+'4') ; n.duration.quarterLength *= sustain/config.beat_resolution
                c.add(n)
            else:
                n = note.Rest() ; n.duration.quarterLength *= sustain/config.beat_resolution

            #n.storedInstrument = instrument.Piano()
            n.offset = i/config.beat_resolution
            music_stream.append(n)
            n.offset = i/config.beat_resolution
            #n.storedInstrument = instrument.Piano()

    return music_stream



##



def main():
    prev_data = pickle_load(config.data_path+'.pk')
    if prev_data: print('> appending data to prev file')
    save_data(preprocess() + (prev_data if prev_data else []))



if __name__ == '__main__':
    main()





##

# EXTENSION: Data Creator

##



default_progressions = [
    [2,5,1,6],
    [1,4,5,6],
    [1,5,4,3],
    # #[1,4,6,2], # major goes to 2
    # [6,3,2,1], # minor goes to its 3 - major ascend
    # [6,1,2,3], # minor goes to its 5 - major descend
    # #[1,5,4,3], # major goes to 3 - major descend
    # [1,3,4,5], # major goes to 5 - major ascend
]

default_voicings = [
    [1,3,5],
    [1,3,7],
    [1,5,7],
]

default_rhythms = [
    'llaa,rrhh - llaa,rrhh - llaa,rrhh - llaa,rrhh',
    #'llar,larr - llar,laaa - llar,larr - llar,laaa',
    #'hhrh,llaa - llaa,rhrr - hhrh,llaa - llaa,rhhh',
]

transpose_to_keys = ['C']#[to for to in note_reverse_dict.values() if to != 'R']


# C_chromatic = {
#     'C' : 0,
#     'C#': 1,
#     'D' : 2,
#     'D#': 3,
#     'E' : 4,
#     'F' : 5,
#     'F#': 6,
#     'G' : 7,
#     'G#': 8,
#     'A' : 9,
#     'A#': 10,
#     'B' : 11,
# }
#
# C_chromatic_reverse = {
#     0: 'C',
#     1: 'C#',
#     2: 'D',
#     3: 'D#',
#     4: 'E',
#     5: 'F',
#     6: 'F#',
#     7: 'G',
#     8: 'G#',
#     9: 'A',
#     10:'A#',
#     11:'B',
# }
#
# intervals = {
#     1: 0,
#     2: 2,
#     3: 4,
#     4: 5,
#     5: 7,
#     6: 9,
#     7: 11,
# }
#
# def traverse_chromatic_interval(note, interval):
#     return C_chromatic_reverse[(C_chromatic[note]+intervals[interval])%12]


C_tonic = {
    'C' : 0,
    'D' : 1,
    'E' : 2,
    'F' : 3,
    'G' : 4,
    'A' : 5,
    'B' : 6,
}

C_tonic_reverse = {
    0: 'C',
    1: 'D',
    2: 'E',
    3: 'F',
    4: 'G',
    5: 'A',
    6: 'B',
}

def traverse_tonic_interval(note, interval):
    interval -=1
    return C_tonic_reverse[(C_tonic[note]+interval)%7]


def create_data(progressions=None,voicings=None,rhythms=None,transposes=None,loop=.5):

    if not progressions:
        progressions = default_progressions
    if not voicings:
        voicings = default_voicings
    if not rhythms:
        rhythms = default_rhythms
    if not transposes:
        transposes = transpose_to_keys

    for p_id,progression in enumerate(progressions):

        progression += progression[:int(len(progression)*loop)]

        for v_id,voicing in enumerate(voicings):

            for r_id,rhythm in enumerate(rhythms):

                rhythm = rhythm.split(' - ')
                rhythm = [bar.replace(',','') for bar in rhythm]
                rhythm += rhythm[:int(len(rhythm)*loop)]

                song_sequence = []

                for root,bar in zip(progression,rhythm):

                    for step in bar:

                        timestep = ''

                        if step == 'l':

                            timestep += traverse_tonic_interval(traverse_tonic_interval('C', root), voicing[0])

                        elif step == 'h':

                            for v in voicing[1:]:
                                timestep += traverse_tonic_interval(traverse_tonic_interval('C', root), v)
                                timestep += ','
                            timestep = timestep[:-1]

                        elif step == 'a':

                            for v in voicing:
                                timestep += traverse_tonic_interval(traverse_tonic_interval('C', root), v)
                                timestep += ','
                            timestep = timestep[:-1]

                        elif step == 'r':

                            timestep += 'R'

                        song_sequence.append(timestep)

                song_stream = convert_to_stream(song_sequence)

                transposed_streams = [song_stream.transpose(interval.Interval(pitch.Pitch('C'),pitch.Pitch(to))) for to in transposes]

                for t_id,transposed_stream in enumerate(transposed_streams):

                    transposed_stream.write('midi', fp=config.samples_folder+f'/p{p_id}-v{v_id}-r{r_id}-t{t_id}.mid')



##



def main2():
    create_data()
    main()


##



