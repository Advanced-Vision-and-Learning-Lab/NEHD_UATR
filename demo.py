# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:15:42 2025
Code modified from: https://github.com/Advanced-Vision-and-Learning-Lab/lightning_template/blob/main/demo.py
"""

import argparse
import glob
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

# PyTorch dependencies
import torch

# Lightning dependencies
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

# Local external libraries
from Datasets.DeepShipDataModule import DeepShipDataModule
from Demo_Parameters import Parameters
from Models.Lightning_Wrapper import Lightning_Wrapper
from Utils.Network_Functions import initialize_models
from Utils.Save_Results import generate_filename

plt.ioff()


def main(args):
    if args.HPRC:
        print("Running on HPRC!")
        torch.set_float32_matmul_precision("medium")
    accuracy_dict = {}

    # Set initial parameters
    Network_parameters = Parameters(args)
    print(f"Starting Experiments for {Network_parameters['intermediate_feature']} + {Network_parameters['base_model']}")
    # Name of dataset
    Dataset = Network_parameters["Dataset"]

    # Number of runs and/or splits for dataset
    num_runs = Network_parameters["Splits"][Dataset]

    Network_parameters["num_classes"] = Network_parameters["num_classes"][Dataset]
    # for split in range(0, num_runs):
    for split in range(0, num_runs):
        # Set seed for reproducibility
        # Set same random seed based on split and fairly compare
        # each approach
        torch.manual_seed(split)
        np.random.seed(split)
        random.seed(split)
        torch.cuda.manual_seed(split)
        torch.cuda.manual_seed_all(split)

        print("Initializing/Finding the model path...")
        filename = generate_filename(Network_parameters, split)
        print(f"Saved Model path: {filename}")

        print("Setting up logger...")
        logger = TensorBoardLogger(
            filename, default_hp_metric=False, version="Training"
        )

        # Remove past events to conserve memory allocation
        log_dir = "{}{}/{}".format(logger.save_dir,
                                   logger.name, logger.version)
        files = glob.glob("{}/{}".format(log_dir, "events.out.tfevents.*"))

        for f in files:
            os.remove(f)
        print("Logger set up.")

        print("Initializing Datasets and Dataloaders...")
        if Dataset == "DeepShip":
            # process_data(sample_rate=Network_parameters['sample_rate'], segment_length=Network_parameters['segment_length'])
            data_module = DeepShipDataModule(
                Network_parameters["data_dir"],
                Network_parameters["batch_size"],
                Network_parameters["num_workers"],
                Network_parameters["pin_memory"],
                sample_rate=Network_parameters["sample_rate"],
                segment_length=Network_parameters["segment_length"],
            )
            print("DeepShip DataModule Initialized")
        else:
            raise ValueError("{} Dataset not found".format(Dataset))

        data_module.prepare_data()
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        print("Dataloaders Initialized.")

        print("Initializing Audio Feature Extractor, Intermediate Model and Base Model.")
        audio_feature_extractor, intermediate_feature, base_model = initialize_models(
            data_module, Network_parameters, filename)

        model_ft = Lightning_Wrapper(
            audio_feature_extractor=audio_feature_extractor,
            intermediate_model=intermediate_feature,
            base_model=base_model,
            num_classes=Network_parameters["num_classes"],
            intermediate_learning_rate=Network_parameters["intermediate_lr"],
            base_model_learning_rate=Network_parameters["base_lr"],
            step_size=Network_parameters["step_size"],
            gamma=Network_parameters["gamma"],
            log_dir=filename,
            label_names=Network_parameters["class_names"],
            average="weighted",
        )
        print("Models Initialized.")
        
        # Create a checkpoint callback to save best model based on val accuracy
        print("Setting up checkpoint callback...")
        checkpoint_callback = ModelCheckpoint(
            filename="best_model", mode="max", monitor="val_accuracy"
        )
        print("Checkpoint callback set up.")


        print("Setting up trainer...")
        trainer = Trainer(
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", patience=Network_parameters["patience"]
                ),
                checkpoint_callback,
                TQDMProgressBar(refresh_rate=100),
            ],
            max_epochs=Network_parameters["num_epochs"],
            enable_checkpointing=Network_parameters["save_results"],
            default_root_dir=filename,
            logger=logger,
        )
        print("Trainer set up.")

        print("Training model...")
        trainer.fit(
            model_ft, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        print("Training completed.")
        model_metrics = {"train_accuracies": model_ft.train_accuracies,
                         "val_accuracies": model_ft.val_accuracies[1:], "train_losses": model_ft.train_losses, "val_losses": model_ft.val_losses[1:]}
        # print(model_metrics)
        with open(f'{filename}model_metrics.pkl', 'wb') as f:
            pickle.dump(model_metrics, f)
            # Print the validation accuracy of the best model
            print('Best model validation accuracy: ',
                  checkpoint_callback.best_model_score.item())
            accuracy_dict[f"Run_{split}"] = checkpoint_callback.best_model_score.item(
            )
        # Print the validation accuracy of the best model
        print(
            "Best model validation accuracy: ",
            checkpoint_callback.best_model_score.item(),
        )
        del model_ft
        torch.cuda.empty_cache()

        print(f"**********Run {str(split + 1)} Finished**********")

    print(f"\n Validation Accuracies: {accuracy_dict}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Neural Handcrafted Features Experiments for DeepShip"
    )
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
        help="Location to save models. (default: Saved_Models/)",
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
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs to train each model for (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Number of epochs to stop training based on validation loss (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloader. (default: 8)",
    )
    parser.add_argument(
        "--HPRC",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Flag to run on HPRC (default: False)",
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
        help="NUmber of Frequency Bins for the feature (default: 192)",
    )
    parser.add_argument(
        "--window_length", type=int, default=6144, help="Window Length (default: 6144)"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
