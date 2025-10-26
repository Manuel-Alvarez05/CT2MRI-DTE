
import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np

# Helper utilities provided by the repo
# - dict2namespace / namespace2dict: convert between dict <-> argparse.Namespace for convenience
# - get_runner: fetches the correct Runner class (e.g., BBDMRunner) from the global registry and instantiates it
from utils import dict2namespace, get_runner, namespace2dict

# Multiprocessing & distributed training with PyTorch
import torch.multiprocessing as mp
import torch.distributed as dist

# Optional logging/metrics service (Weights & Biases)
import wandb


# -----------------------------
# 1) ARGUMENTS + CONFIG LOADING
# -----------------------------

def parse_args_and_config():
    """Parse CLI args, load YAML, apply overrides, and set up logging config.

    Returns
    -------
    namespace_config : argparse.Namespace
        Hierarchical configuration in Namespace form (easy attr access), after applying CLI overrides.
    dict_config : dict
        Same configuration as a plain dict (useful to log to W&B or save to disk).
    """

    # Create an argument parser with an optional description (from module docstring if available)
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # ---- Basic run controls ----
    parser.add_argument('-c', '--config', type=str, default='BB_base.yml', help='Path to the YAML config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('-r', '--result_path', type=str, default='results', help='Where to save results/checkpoints')

    # ---- Mode flags ----
    parser.add_argument('-t', '--train', action='store_true', default=False, help='Train if set; otherwise test')
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='Run sampling/evaluation pipeline')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='Do a sample pass right when starting (debug)')
    parser.add_argument('--save_top', action='store_true', default=False, help='Save top‑k models by validation metric')

    # ---- System / hardware ----
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU list: "0" or "0,1,2"; use -1 for CPU')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port (change if conflicts)')

    # ---- Resume from checkpoints ----
    parser.add_argument('--resume_model', type=str, default=None, help='Path to a model checkpoint to resume from')
    parser.add_argument('--resume_optim', type=str, default=None, help='Path to an optimizer/scheduler checkpoint')

    # ---- Naming ----
    parser.add_argument('--exp_name', type=str, help='Experiment name (subfolder under results/)')

    # ---- Training schedule overrides ----
    parser.add_argument('--max_epoch', type=int, default=None, help='Override: total epochs')
    parser.add_argument('--max_steps', type=int, default=None, help='Override: total steps (alternative to epochs)')

    # ---- Data overrides ----
    parser.add_argument('--dataset_type', type=str, default='', help='Override dataset type when sampling')
    parser.add_argument('--plane', type=str, help='Input plane: axial, sagittal, or coronal')
    parser.add_argument('--HW', type=int, default=128, help='Image size (H=W); also passed to the UNet')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for train/val/test')

    # ---- Model / diffusion overrides ----
    parser.add_argument('--mt_type', type=str, default=None, help='Model type variant (if supported)')
    parser.add_argument('--objective', type=str, default=None, help='Training objective (e.g., predict noise/velocity)')
    parser.add_argument('--loss_type', type=str, default=None, help='Loss function type (e.g., mse)')
    parser.add_argument('--sample_step', type=int, default=None, help='Sampler step interval')
    parser.add_argument('--num_timesteps', type=int, default=None, help='Number of diffusion timesteps in sampling')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM stochasticity parameter (0 = deterministic)')
    parser.add_argument('--max_var', type=float, default=1.0, help='Max variance cap for noise schedule')
    parser.add_argument('--inference_type', type=str, default=None,
                        help='Sampler variant: normal, average, ISTA_average, ISTA_mid')
    parser.add_argument('--ISTA_step_size', type=float, default=None, help='Step size for ISTA inference variants')
    parser.add_argument('--num_ISTA_step', type=int, default=None, help='Number of ISTA steps at inference')

    # Parse all command‑line arguments now
    args = parser.parse_args()

    # Load the YAML config file specified by -c/--config
    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert dict → Namespace for easy attribute access (config.a.b instead of config['a']['b'])
    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args  # keep original CLI args around

    # ---- Apply CLI overrides so they take effect even if YAML has defaults ----
    # WHY: quick experiments without editing the YAML each time.
    namespace_config.data.dataset_name += f"_{args.HW}"
    namespace_config.data.dataset_config.image_size = args.HW
    namespace_config.data.train.batch_size = args.batch
    namespace_config.data.val.batch_size   = args.batch
    namespace_config.data.test.batch_size  = args.batch

    namespace_config.model.model_name = args.exp_name
    namespace_config.model.BB.params.UNetParams.image_size = args.HW
    namespace_config.model.BB.params.eta      = args.ddim_eta
    namespace_config.model.BB.params.max_var  = args.max_var

    # Optional: resume paths & training length overrides
    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps

    # Optional: model behavior overrides
    if args.mt_type is not None:
        namespace_config.model.BB.params.mt_type = args.mt_type
    if args.objective is not None:
        namespace_config.model.BB.params.objective = args.objective
    if args.loss_type is not None:
        namespace_config.model.BB.params.loss_type = args.loss_type
    if args.sample_step is not None:
        namespace_config.model.BB.params.sample_step = args.sample_step
    if args.num_timesteps is not None:
        namespace_config.model.BB.params.num_timesteps = args.num_timesteps
    if args.inference_type is not None:
        namespace_config.model.BB.params.inference_type = args.inference_type
    if args.ISTA_step_size is not None:
        namespace_config.model.BB.params.ISTA_step_size = args.ISTA_step_size
    if args.num_ISTA_step is not None:
        namespace_config.model.BB.params.num_ISTA_step = args.num_ISTA_step

    # If the user wants to sample/evaluate, let them optionally choose a dataset type at CLI
    if args.sample_to_eval:
        if args.dataset_type != '':
            namespace_config.data.dataset_type = args.dataset_type

    # Convert back to a dict (useful for logging to W&B)
    dict_config = namespace2dict(namespace_config)

    # Prepare a compact dict with the most important knobs for experiment tracking
    config_dict = {
        "train": args.train,
        "gpu_ids": args.gpu_ids,
        "dataset_type": namespace_config.data.dataset_type,
        "image_size": namespace_config.data.dataset_config.image_size,
        "max_epoch": namespace_config.training.n_epochs,
        "n_steps": namespace_config.training.n_steps,
        "batch_size": namespace_config.data.train.batch_size,
        "mt_type": namespace_config.model.BB.params.mt_type,
        "objective": namespace_config.model.BB.params.objective,
        "loss_type": namespace_config.model.BB.params.loss_type,
        "sample_step": namespace_config.model.BB.params.sample_step,
        "num_timesteps": namespace_config.model.BB.params.num_timesteps,
        "ddim_eta": namespace_config.model.BB.params.eta,
        "max_var": namespace_config.model.BB.params.max_var,
    }

    # Initialize Weights & Biases only for training runs
    if not args.sample_to_eval:
        try:
            # If project/entity are empty, wandb.init may fail; the except keeps training running anyway
            wandb.init(project="", entity="", name=namespace_config.model.model_name, config=config_dict)
        except Exception:
            print('Could not init wandb')

    return namespace_config, dict_config


# -----------------------------
# 2) REPRODUCIBILITY UTILITIES
# -----------------------------

def set_random_seed(SEED=1234):
    """Set seeds across Python, NumPy, and PyTorch to make runs reproducible.

    Notes
    -----
    - cudnn.benchmark=False and cudnn.deterministic=True aim for determinism (may reduce speed slightly).
    - In DDP, each process calls this, so all ranks share the same seed base.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------------------------
# 3) DISTRIBUTED (DDP) TRAIN/TEST FUNCTION
# ----------------------------------------

def DDP_run_fn(rank, world_size, config):
    """Function executed by each spawned DDP process.

    Parameters
    ----------
    rank : int
        Local rank / process index (0..world_size-1).
    world_size : int
        Total number of processes.
    config : argparse.Namespace
        Full configuration (copied into each process).
    """
    # Required master address/port for processes to rendezvous
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.args.port

    # Initialize the default process group using NCCL (fast GPU backend)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Make results as reproducible as possible
    set_random_seed(config.args.seed)

    # Each process selects a different GPU based on its rank
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    # Expose device info to the rest of the code via config
    config.training.device = [torch.device(f"cuda:{local_rank}")]
    print('using device:', config.training.device)
    config.training.local_rank = local_rank

    # Build the correct Runner (e.g., BBDMRunner) from the registry and run
    runner = get_runner(config.runner, config)

    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


# -------------------------------------
# 4) CPU or SINGLE‑GPU LAUNCHER (shared)
# -------------------------------------

def CPU_singleGPU_launcher(config):
    """Shared path for CPU or single‑GPU execution (no DDP)."""
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


# --------------------------------------
# 5) DDP LAUNCHER (spawns N processes)
# --------------------------------------

def DDP_launcher(world_size, run_fn, config):
    """Spawn `world_size` processes and run `run_fn` (usually DDP_run_fn) on each.

    - copy.deepcopy(config) ensures each child gets an isolated config object.
    - join=True blocks until all processes have finished.
    """
    mp.spawn(run_fn,
             args=(world_size, copy.deepcopy(config)),
             nprocs=world_size,
             join=True)


# ---------------
# 6) MAIN ENTRY
# ---------------

def main():
    # 6.1 Parse config and args (now we know what to do and with what settings)
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args

    # 6.2 Decide execution mode based on --gpu_ids
    gpu_ids = args.gpu_ids

    if gpu_ids == "-1":  # CPU mode
        nconfig.training.use_DDP = False
        nconfig.training.device = [torch.device("cpu")]
        CPU_singleGPU_launcher(nconfig)

    else:
        # If user passed a comma‑separated list (e.g., "0,1,2"), we go DDP; otherwise single GPU
        gpu_list = gpu_ids.split(",")
        if len(gpu_list) > 1:
            # Limit visible GPUs to the user selection, then launch DDP
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            nconfig.training.use_DDP = True
            DDP_launcher(world_size=len(gpu_list), run_fn=DDP_run_fn, config=nconfig)
        else:
            # Single GPU path
            nconfig.training.use_DDP = False
            nconfig.training.device = [torch.device(f"cuda:{gpu_list[0]}")]
            CPU_singleGPU_launcher(nconfig)
    return


if __name__ == "__main__":
    main()
