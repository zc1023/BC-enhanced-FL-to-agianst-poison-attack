import argparse
import torch

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def create_argparser():
    defaults_clients_num = dict(
        validation_nodes_num = 0,
        clients_num = 10,
        benign_clients_num = 0,
        flipping_attack_num = 0,
        grad_zero_num = 0,
        grad_scale_num = 0,
        backdoor_num = 0,
    )

    defaults_clients_args = dict(
        flip_malicous_rate=1.0,
        grad_zero_rate=0.5,
        grad_scale_rate=0.5,
    )

    defaults = dict(
        datasets = 'CIFAR10',
        project_name = 'BC-enhanced-FL-to-against-poison-attack-2024',
        model = 'Cifar10CNN',
        seed = 10,
        batch_size = 64,
        epoch_num = 100,
        local_epoch_num = 2,
        data_type = 'iid',
        optimizer = 'sgd',
        lr = 1e-2,
        eva_type = "ACC", #ACC,FSAR,BSAR
        aggregate_type = "trimmed_mean", #fedavg, krum, median, trimmed_mean
        wandb_log = False,
        wandb_resume = False,
    )
    defaults.update(defaults_clients_args)
    defaults.update(defaults_clients_num)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser=parser,default_dict=defaults)
    return parser

if __name__ =='__main__':
    args = create_argparser().parse_args()
    print(args)