# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:20:44 2019
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""


def Parameters(args, learn_hist=True, learn_edge_kernels=True, feature_init=True, learn_transform=True, dilation=1, mask_size=[3, 3]):

    # Flag for if results are to be saved out
    # Set to True to save results out and False to not save results
    save_results = args.save_results

    # Location to store trained models
    # Always add slash (/) after folder name
    folder = args.folder

    intermediate_features = {"1" : "nehd", "2": "ehd", "3": "histogram", "4": None}

    base_models = {"1" : "linear", "2": "resnet50", "3": "pann", "4": "vit", "5": "ast"}

    intermediate_feature = intermediate_features[args.intermediate_feature]
    if intermediate_feature == "ehd":
        is_ehd = True
    else:
        is_ehd = False
    
    base_model = base_models[args.base_model]
    if base_model == "pann":
        feature_ext_padding = [12, 12, 4, 4]
    else:
        feature_ext_padding = [1, 1, 0, 0]

    # Select aggregation type for layer: 'Local' or 'GAP'.
    # Recommended is RBF (implements histogram function in paper)
    aggregation_type = "Local"

    # Flags to learn histogram parameters (bins/centers) and spatial masks

    mask_size = mask_size
    window_size = [5, 5]
    angle_res = 45

    learn_hist = True
    learn_edge_kernels = True
    feature_init = True
    learn_transform = True

    normalize_count = True
    normalize_kernel = True  # Need to be normalized for histogram layer (maybe b/c of hist initialization)
    threshold = 1 / int(360 / angle_res)  # 10e-3 #1/int(360/angle_res) #.9

    stride = 2
    dilation = 1

    # Number of bins for histogram layer. Recommended values are the number of
    # different angle resolutions used (e.g., 3x3, 45 degrees, 8 orientations) or LBP (user choice).

    num_bins = int(360 / angle_res)
    out_channels = num_bins + 1

    # Set learning rate for model
    # Recommended values are .001 and .01
    intermediate_lr = args.intermediate_lr
    base_lr = args.base_lr

    # Set whether to enforce sum to one constraint across bins (default: True)
    # Needed for EHD feature (softmax approximation of argmax)
    normalize_bins = True

    # Set step_size and decay rate for scheduler
    # In paper, learning rate was decayed factor of .1 every ten epochs (recommended)
    step_size = 1000
    gamma = 0.1

    # Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    # training batch size is recommended to be 64. If using at least two GPUs,
    # the recommended training batch size is 128 (as done in paper)
    # May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {"train": args.train_batch_size, "val": args.val_batch_size, "test": args.test_batch_size}

    # Pin memory for dataloader (set to True for experiments)
    pin_memory = True

    # Set number of workers, i.e., how many subprocesses to use for data loading.
    # Usually set to 0 or 1. Can set to more if multiple machines are used.
    # Number of workers for experiments for two GPUs was three
    num_workers = args.num_workers


    # Visualization parameters for figures
    fig_size = 16
    font_size = 30


    data_selection = 1
    Dataset_names = {1:'DeepShip'} 
    Dataset = Dataset_names[data_selection]
    Data_dirs = {'DeepShip': './Datasets/DeepShip/Segments_3s_16000hz/'}

    num_classes = {"DeepShip": 4}
    Class_names = {"DeepShip": ["Cargo", "Passengership", "Tanker", "Tug"]}
    data_dir = Data_dirs[Dataset]
    Class_names = Class_names[Dataset]
    Splits = {'DeepShip': 3}
    # Audio Feature Parameters
    # segment_length = 3
    segment_length = 30

    sample_rate = 16000
    window_length = args.window_length
    freq_bins = args.freq_bins
    hop_length = args.hop_length
    # folder += f"lr_{lr_dict[str(lr)]}/"
    folder += f"w_{window_length}_h_{hop_length}_f_{freq_bins}/"
    # folder += f"w_{window_length}_h_{hop_length}_f_{freq_bins}_m_{mel_freq_bins}/"

    # Adjust the number of channels depending on the dataset and fusion method

    in_channels = 1
    ###############

    num_epochs = args.num_epochs
    patience = args.patience
    HPRC = args.HPRC
    audio_feature = args.audio_feature
    figsize = (12, 12)
    fontsize = 12

    # Return dictionary of parameters
    Network_parameters = {
        "save_results": save_results,
        "folder": folder,
        "Dataset": Dataset,
        'Splits': Splits,
        "data_dir": data_dir,
        "num_workers": num_workers,
        "base_lr": base_lr,
        "intermediate_lr": intermediate_lr,
        "step_size": step_size,
        "gamma": gamma,
        "batch_size": batch_size,
        "mask_size": mask_size,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "normalize_count": normalize_count,
        "normalize_bins": normalize_bins,
        "num_bins": num_bins,
        "num_classes": num_classes,
        "pin_memory": pin_memory,
        "aggregation_type": aggregation_type,
        "window_size": window_size,
        "angle_res": angle_res,
        "threshold": threshold,
        "stride": stride,
        "is_ehd" :is_ehd,
        "feature_init": feature_init,
        "learn_transform": learn_transform,
        "fig_size": fig_size,
        "font_size": font_size,
        "normalize_kernel": normalize_kernel,
        "learn_hist": learn_hist,
        "learn_edge_kernels": learn_edge_kernels,
        "class_names": Class_names,
        "dilation": dilation,
        "figsize": figsize,
        "fontsize": fontsize,
        "num_epochs":num_epochs,
        "patience": patience,
        "HPRC": HPRC,
        "segment_length": segment_length,
        "sample_rate": sample_rate,
        "window_length": window_length,
        "freq_bins": freq_bins,
        "hop_length": hop_length,
        "base_model": base_model,
        "audio_feature": audio_feature,
        "intermediate_feature": intermediate_feature,
        "feature_ext_padding": feature_ext_padding,
    }

    return Network_parameters
