
import getpass
import os
import shutil
import socket
import sys
import tempfile
import warnings
from contextlib import contextmanager

import wandb

from runner import Runner
from utilities import GeneralUtility

warnings.filterwarnings('ignore')

debug = "--debug" in sys.argv

defaults = dict(
    # System
    seed=1,

    # Data
    dataset='ai4forest_debug',
    batch_size=5,

    # Architecture
    arch='unet',  # Defaults to unet
    backbone='resnet50',  # Defaults to resnet50
    use_pretrained_model=False,

    # Optimization
    optim='AdamW',  # Defaults to AdamW
    loss_name='shift_huberf',  # Defaults to shift_l1
    n_iterations=100,
    log_freq=5,
    initial_lr=1e-3,
    weight_decay=1e-2,
    use_standardization=False,
    use_augmentation=False,
    use_label_rescaling=False,

    # Efficiency
    fp16=False,
    use_memmap=False,
    num_workers_per_gpu=8,   # Defaults to 8

    # Other
    use_weighted_sampler='g10',
    use_weighting_quantile=10,
    use_swa=False,
    use_mixup=False,
    use_grad_clipping=False,
    use_input_clipping=False,   # Must be in [False, None, 1, 2, 5]
    n_lr_cycles=0,
    cyclic_mode='triangular2',
    )

if not debug:
    # Set everything to None recursively
    defaults = GeneralUtility.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = GeneralUtility.update_config_with_default(config, defaults)


@contextmanager
def tempdir():
    username = getpass.getuser()
    tmp_root = '/scratch/local/' + username
    tmp_path = os.path.join(tmp_root, 'tmp')
    if os.path.isdir('/scratch/local/') and not os.path.isdir(tmp_root):
        os.mkdir(tmp_root)
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path): os.mkdir(tmp_path)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        assert 'htc-' not in os.uname().nodename, "Not allowed to write to /tmp on htc- machines."
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    # Check if we are running on the GCP cluster, if so, mark as potentially preempted
    is_htc = 'htc-' in os.uname().nodename
    is_gcp = 'gpu' in os.uname().nodename and not is_htc
    if is_gcp:
        print('Running on GCP, marking as preemptable.')
        wandb.mark_preempting()  # Note: This potentially overwrites the config when a run is resumed -> problems with tmp_dir

    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
