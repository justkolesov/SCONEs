import argparse
import pathlib
import traceback
import shutil
import logging
import yaml
import sys
import torch
import numpy as np
import copy
from scones.runners import *
import util
import os
from scones.runners import GaussianRunner

def parse_args_and_config():
    
    
    #========PARSE_BLOCK======#
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--dry_run', action="store_true", help="If true, no code is excuted and the script is dry-run.")
    parser.add_argument('--overwrite', action="store_true", help="If true, automatically overwrite without asking.")
    #parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--save_labels', action="store_true", help="If set to true then sampling will also save labels and bproj details of source.")
    args = parser.parse_args()
     
    with open(os.path.join(args.config), 'r') as f:
        config = yaml.load(f)
    new_config = util.dict2namespace(config)
    
    p = pathlib.Path(args.config)
    scones_path = pathlib.Path(*p.parts[:-2])
    new_config.log_path = os.path.join( scones_path, new_config.logging.exp, 'logs', new_config.logging.doc)
    print(util.magenta("Parse block is done!"))
    #=========================#
    
    
    #=========================#
    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    level = getattr(logging, new_config.logging.verbose_stderr.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(new_config.logging.verbose_stderr))
    handler1.setLevel(level)
    logger.addHandler(handler1)
    level = getattr(logging, new_config.logging.verbose_logger.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(new_config.logging.verbose_logger))
    logger.setLevel(level)
    print(util.yellow("logger is done!"))
    #===========================#

    os.makedirs(os.path.join(scones_path,new_config.logging.exp, 'image_samples'), exist_ok=True)
    new_config.image_folder = os.path.join(scones_path, new_config.logging.exp, 'image_samples', new_config.logging.image_folder)
    if not os.path.exists(new_config.image_folder):
        os.makedirs(new_config.image_folder)
        print(util.magenta('hui'))
        if(args.save_labels):
            os.makedirs(os.path.join(new_config.image_folder, "labels"))
    else:
        #overwrite = args.overwrite
        print(util.magenta('pizda'))
        if(not overwrite):
            response = input("Image folder already exists. Overwrite? (Y/N) ")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(new_config.image_folder)
            os.makedirs(new_config.image_folder)
            if (args.save_labels):
                os.makedirs(os.path.join(new_config.image_folder, "labels"))
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

            
    #========DEVICE======#
    # add device
    device = torch.device(f'cuda:{new_config.n_gpu}') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device
    print(util.magenta("device is ready!"))
    #========DEVICE======#

    # todo: rethink the config design to avoid translation
    new_config.ncsn.device = device
    new_config.compatibility.device = device

    
    # set random seed
    torch.manual_seed(new_config.seed)
    np.random.seed(new_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(new_config.seed)

    torch.backends.cudnn.benchmark = True
    print(util.magenta("seed is set!"))
    return args, new_config


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(new_config.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(new_config.logging.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    if(args.dry_run):
        print("Dry run successful!")
        print(f"GPU Availability: {torch.cuda.is_available()}")
    else:
        try:
            if(config.target.data.dataset.upper() in ["GAUSSIAN", "GAUSSIAN-HD"]):
                runner = GaussianRunner(config)
            else:
                runner = SCONESRunner(args, config)
                
            runner.sample()
        except:
            logging.error(traceback.format_exc())

    return 0

if __name__ == '__main__':
    sys.exit(main())