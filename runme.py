import config

from ext import cls


def main():

    ctr = 1

    while ctr:

        inp = input(f'Select: \n\t1 Configure \n\t2 Train \n\t3 Interact \n\n      > ')
        cls()

        if inp == '1':
            list_config()
            change_config()

        elif inp == '2':
            import train
            train.main()

        elif inp == '3':
            import interact
            interact.main()

        elif inp == '0':
            cls()
            break

        cls()
        ctr +=1


def list_config():

    print('Config Params:')

    for i,e in enumerate(dir(config)):

        print(f'\t{i} {e}')


def change_config():

    print(
    '''
    Warning:
    IF running parallel training, AND modifying Neural Params ,
    code changes @ config.py ,(changes through runtime will not be seen.)
    ''')

    while 1:

        if (i := input('Enter param index: ')) == '':
            break
        else:
            try:
                index = int(i)
                print(f'> {dir(config)[index]} = {getattr(config,dir(config)[index])}')
            except:
                try:
                    index = dir(config).index(i)
                    print(f'> {index} = {getattr(config,i)}')
                except Exception as e:
                    print(f'Error: execution failed {e}')
                    continue
            if (e := input('Enter param value: ')) != '':
                try:
                    # exec(f'config.{dir(config)[index]} = {eval(e)}')
                    setattr(config, dir(config)[index],eval(e))
                except Exception as e1:
                    try:
                        # exec(f'config.{dir(config)[index]} = {e}')
                        setattr(config, dir(config)[index], e)
                    except Exception as e2:
                        print(f'Error: execution failed {e1} & {e2}')


##


if __name__ == '__main__':

    cls()

    try:
        import torch
        import qiskit
        import pretty_midi
        import music21
        import matplotlib
    except:
        print('Warning: one or more required pip packages cannot be found, proceeding with installation..')
        
        from os import system

        try:
            from torch import __version__ as v
            if v != '1.5.0':
                if input(f'Warning: torch version {v} found, but strictly requires 1.5.0 to run, \nproceed with the switch ? (you can always run pip update torch later) (y/n): ').lower() == 'y':
                    system('pip -y uninstall torch')
                    system('pip3 -y uninstall torch')
                    assert None
        except:
            print('installing torch..')
            system('pip install torch==1.5.0')
            system('pip3 install torch==1.5.0')

        try: import qiskit
        except:
            print('installing qiskit..')
            system('pip install qiskit')
            system('pip3 install qiskit')

        try: import music21
        except: 
            print('installing music21..')
            system('pip install music21==5.7.2')
            system('pip3 install music21==5.7.2')

        try: import pretty_midi
        except:
            print('installing pretty-midi..')
            system('pip install pretty_midi')
            system('pip3 install pretty_midi')

        try: import matplotlib
        except:
            print('installing matplotlib..')
            system('pip install matplotlib')
            system('pip3 install matplotlib')
            
    # try: from qiskit import IBMQ; IBMQ.load_account()
    # except: print('Warning: IBMQ not accessible, run offline only')

    main()