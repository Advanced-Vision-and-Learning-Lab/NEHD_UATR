# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:15:42 2025
Code modified from: https://github.com/Advanced-Vision-and-Learning-Lab/lightning_template/blob/main/View_Results.py
"""


## Python standard libraries
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from Datasets.DeepShipDataModule import DeepShipDataModule
from Models.Lightning_Wrapper import Lightning_Wrapper
import glob

## PyTorch Lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger


## PyTorch dependencies
import torch
## Local external libraries
from Demo_Parameters import Parameters
from Utils.Network_Functions import initialize_models
from Utils.Save_Results import generate_filename, aggregate_tensorboard_logs, aggregate_and_visualize_confusion_matrices


plt.ioff()

def main(args):    
    print(
        f"Starting Experiments"
    )

    if args.HPRC:
        print("Running on HPRC!")
        torch.set_float32_matmul_precision("medium")

    # Set initial parameters
    Network_parameters = Parameters(args)

    # Name of dataset
    Dataset = Network_parameters["Dataset"]

    # Number of runs and/or splits for dataset
    num_runs = Network_parameters["Splits"][Dataset]

    Network_parameters["num_classes"] = Network_parameters["num_classes"][Dataset]
    label_names = Network_parameters['class_names']
    phases = args.phases

    # for split in range(0, num_runs):
    for split in range(0, num_runs):
        torch.manual_seed(split)
        np.random.seed(split)
        np.random.seed(split)
        torch.cuda.manual_seed(split)
        torch.cuda.manual_seed_all(split)
        torch.manual_seed(split)
        
        # Get a filename for saving results
        print("Initializing the model path...")   
        sub_dir = generate_filename(Network_parameters, split)
        
        if not os.path.isdir(sub_dir):
            print(f"Model for split {split} not found.")
            break
        
        checkpt_path = os.path.join(sub_dir, 'lightning_logs/Training/checkpoints/best_model.ckpt')  
        filename = generate_filename(Network_parameters,split)
        print("Model path: ", filename)

        # Set up the logger
        print("Setting up logger...")
        logger = TensorBoardLogger(filename, version = 'Val_Test', default_hp_metric=False)
        
        #Remove past events to conserve memory allocation
        log_dir = '{}{}/{}'.format(logger.save_dir,logger.name,logger.version)
        files = glob.glob('{}/{}'.format(log_dir,'events.out.tfevents.*'))
        
        for f in files:
            os.remove(f)
        print("Logger set up.")
        
        #Initializing Data Module
        if Dataset == 'DeepShip':
            data_module = DeepShipDataModule(Network_parameters['data_dir'],Network_parameters['batch_size'],
                                            Network_parameters['num_workers'], Network_parameters['pin_memory'],
                                            sample_rate=Network_parameters['sample_rate'], segment_length=Network_parameters['segment_length'])
            print('DeepShip DataModule Initialized')
        else:
            raise ValueError('{} Dataset not found'.format(Dataset))
        


        print("Initializing Datasets and Dataloaders...")                
            
        # Initialize the data loaders
        print("Preparing data loaders...")
        data_module.prepare_data()
        data_module.setup("fit")
        val_loader = data_module.val_dataloader()
        data_module.setup("test")
        test_loader = data_module.test_dataloader()
        print("Dataloaders Initialized.")


        audio_feature_extractor, intermediate_feature, base_model = initialize_models(
        data_module, Network_parameters, filename)
        print("Model Initialized.")
        
        # Load model
        print('Initializing model as Lightning Module...')
        model = Lightning_Wrapper.load_from_checkpoint( # TODO: Implement map parameter since this is likely useful
            checkpoint_path=checkpt_path, # TODO: Decide how to deal with multiple versions
            # map_location = 
            hparams_file=os.path.join(sub_dir, 'lightning_logs/Training/checkpoints/hparams.yaml'),
        audio_feature_extractor=audio_feature_extractor,
        intermediate_model=intermediate_feature,
        base_model=base_model,
        num_classes=Network_parameters["num_classes"], strict=True, logger=logger,
            log_dir = filename, label_names = label_names)
        print('Model initialized as Lightning Module...')


        # Create a checkpoint callback
        print("Setting up checkpoint callback...")
        checkpoint_callback = ModelCheckpoint(filename = 'best_model',mode='max',
                                            monitor='val_accuracy')
        print("Checkpoint callback set up.")

        # Train and evaluate
        print("Setting up trainer...")
        trainer = Trainer(callbacks=[checkpoint_callback], 
                        enable_checkpointing = Network_parameters['save_results'], 
                        default_root_dir = filename, logger=logger) # forcing it to use CPU
        print("Trainer set up.")
        
        # Validation
        trainer.validate(model, dataloaders = val_loader)
        
        # Test
        trainer.test(model, dataloaders = test_loader)

    
        print('**********Run ' +  str(split + 1) + '  Finished**********')
        
    print('Getting aggregated results...')
    sub_dir = os.path.dirname(sub_dir.rstrip('/'))
    
    aggregation_folder = 'Aggregated_Results/'
    aggregate_and_visualize_confusion_matrices(sub_dir, aggregation_folder, 
                                            dataset=Dataset,label_names = label_names,
                                            figsize=Network_parameters['figsize'], fontsize=Network_parameters['fontsize'])
    aggregate_tensorboard_logs(sub_dir, aggregation_folder,Dataset)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate results from experiments')
  
    parser.add_argument(
        "--save_results",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save results of experiments (default: True)",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="Saved_Models/", 
        help="Location to save models. (default: Saved_Models)",
    )
    parser.add_argument(
        "--intermediate_feature",
        type=str,
        default="1",
        help="Select feature to evaluate 1. NEHD, 2. EHD, 3. Histogram, 4. None. (default: 1)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="1",
        help="Select feature to evaluate 1. Linear Layer, 2. resnet50, 3. PANN, 4. ViT, 5. AST. (default: 1)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,  
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=128, 
        help="input batch size for validation (default: 128)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128, 
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloader. (default: 1)",
    )
    parser.add_argument(
        "--intermediate_lr",
        type=float,
        default=0.001, 
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.001,  
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--HPRC",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Flag to run on HPRC (default: False)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to stop training based on validation loss (default: 10)",
    )
    parser.add_argument(
        "--audio_feature",
        type=str,
        default="STFT",
        help="Audio Spectrogram to use (default: STFT)",
    )
    parser.add_argument(
        "--hop_length", type=int, default=4096, help="Hop Length (default: 4096)"
    )
    parser.add_argument(
        "--freq_bins",
        type=int,
        default=192,
        help="Number of Frequency Bins for the feature (default: 192)",
    )
    parser.add_argument(
        "--window_length", type=int, default=6144, help="Window Length (default: 6144)"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)