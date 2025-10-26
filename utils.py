
import argparse          # Used to create Namespace objects for cleaner config access
import importlib         # Lets us import Python modules and classes dynamically
import omegaconf.dictconfig  # Handles OmegaConf DictConfig objects (used in YAML parsing)

# Import the global registry that stores references to all registered datasets and runners
from Register import Registers

# Import a specific runner so it's registered when utils is imported
# (This line ensures the BBDMRunner class is loaded and added to Registers.runners)
from runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner


# ----------------------------------------------------------
# 1) Convert configuration dictionary → argparse.Namespace
# ----------------------------------------------------------

def dict2namespace(config):
    """Recursively convert a dictionary (and nested dicts) into a Namespace object.

    Why: Using config.model.param instead of config['model']['param'] is cleaner and less error‑prone.
    This is useful when you load YAML configs that are naturally dictionaries.
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        # If the value is another dict or an OmegaConf DictConfig, recurse deeper
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# ----------------------------------------------------------
# 2) Convert argparse.Namespace → configuration dictionary
# ----------------------------------------------------------

def namespace2dict(config):
    """Reverse of dict2namespace: turn Namespace back into a dictionary.

    Why: Needed for saving logs, exporting configs, or initializing wandb (which expects dicts).
    """
    conf_dict = {}
    for key, value in vars(config).items():
        # Recurse if the value itself is another Namespace
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict


# ----------------------------------------------------------
# 3) Dynamic class import from string
# ----------------------------------------------------------

def get_obj_from_str(string, reload=False):
    """Import a Python class or function from a string like 'module.submodule.ClassName'.

    Example
    -------
    >>> obj = get_obj_from_str('torch.optim.Adam')
    >>> print(obj)
    <class 'torch.optim.adam.Adam'>

    Parameters
    ----------
    string : str
        Full module path and class name, separated by a dot.
    reload : bool, optional
        If True, the module will be reloaded (useful during code debugging).
    """
    module, cls = string.rsplit(".", 1)  # Split 'package.module.Class' → ['package.module', 'Class']

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    # Import the module and return the attribute/class requested
    return getattr(importlib.import_module(module, package=None), cls)


# ----------------------------------------------------------
# 4) Instantiate an object directly from a config dict
# ----------------------------------------------------------

def instantiate_from_config(config):
    """Create an object instance from a config dictionary.

    The config must contain:
    - 'target': full path to the class (e.g., 'torch.optim.Adam')
    - 'params': optional dict of keyword arguments for the class constructor

    Example
    -------
    cfg = { 'target': 'torch.optim.Adam', 'params': {'lr': 1e-3} }
    opt = instantiate_from_config(cfg)
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")

    # Fetch the class/function and call it with its params
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# ----------------------------------------------------------
# 5) Get the right Runner from the Registry
# ----------------------------------------------------------

def get_runner(runner_name, config):
    """Return the correct Runner class instance by name.

    The runner (like 'BBDMRunner') must already be registered inside Registers.runners.
    This function simply looks it up and creates an instance with the given config.
    """
    runner = Registers.runners[runner_name](config)
    return runner
