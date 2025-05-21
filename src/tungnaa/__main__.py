import sys
import fire

def help():
    print("""
    available subcommands:
        run: run tungnaa instrument
        devices: list audio devices
        prep: data preprocessing for training
        trainer: run a new tungnaa training (GPU recommended)
        resume: create a Trainer from checkpoint
    """)

def _main():
    try:
        cmd = sys.argv[1]
        fire_args = sys.argv[2:]
    except IndexError:
        help()
        exit(0)

    if cmd == 'run':
        from tungnaa.gui.qtgui import main as gui
        fire.Fire(gui, fire_args, "tungnaa run")

    elif cmd == 'devices':
        from tungnaa.gui.backend import print_audio_devices
        print_audio_devices()

    elif cmd == 'prep':
        from tungnaa.prep import main as prep
        fire.Fire(prep, fire_args, "tungnaa prep")

    elif cmd == 'trainer':
        from tungnaa.train import Trainer
        fire.Fire(Trainer, fire_args, "tungnaa trainer")

    elif cmd == 'resume':
        from tungnaa.train import resume
        fire.Fire(resume, fire_args, "tungnaa resume")

    elif cmd in ['list-models', 'list_models']:
        from tungnaa.gui.downloads import main
        fire.Fire(main, fire_args)

    else:
        help() 

if __name__=='__main__':
    _main()