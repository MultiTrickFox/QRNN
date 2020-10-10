##

from math import log2, ceil


## Misc Params ##

act_classical_rnn  = False

reconstruct_qstate = True


## Neural Params ##

timestep_size  = 13

hm_qbits       = ceil(log2(timestep_size))
statevec_size  = 2**hm_qbits

in_size        = timestep_size  # timestep_size+statevec_size

ansatz_mode    = 0 # 0 for Mottonen, 1 for U3s
entangle_mode  = -1 # -1 for off, 0 for weak, 1 for strong

circuit_layers = 1

out_size       = (hm_qbits*3 if ansatz_mode == 1 else statevec_size-1) *circuit_layers if not act_classical_rnn else in_size

creation_info  = [in_size, 'g', 32, 'f2' if not act_classical_rnn else 'f1', out_size]

init_xavier    = True

forget_bias    = 0


## Data Params ##

    # pre-process
samples_folder  = 'data'
data_path       = 'data'

polyphony       = False

combine_instrus = True

beat_resolution = 2 # 8 # 4

    # post-process
hm_bars_grouped = 4 # 4 # 2
hm_bars_slide   = hm_bars_grouped/2

dev_ratio       = .2

shuffle_split   = False
shuffle_epoch   = True


## Training Params ##

hm_bars_teacher = hm_bars_grouped # hm_bars_grouped/2

fresh_model     = True
fresh_meta      = True

learning_rate   = 1e-2 # 5e-2
hm_epochs       = 100
batch_size      = 50

loss_squared    = False

gradient_clip   = 0 # 0 for off

optimizer       = 'custom'

train_parallel  = False
train_combined  = True

initialize_loss = True

model_save_path = 'model'

disp_batches    = False

ckp_save_epochs = []

overwrite_model = False


## Generation Params ##

model_load_path = model_save_path

note_pick_mode  = 3 # 0 for theoretical highest, 1 for experimental highest, 2 for experimental polyphonic, 3 for experimental monophonic

preferred_backend = None

##

