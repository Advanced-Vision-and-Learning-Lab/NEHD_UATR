"""
Created on Tue Aug  6 10:26:08 2024
@author: aagashe
"""
from Models.RBFHistogramPooling import HistogramLayer
from Utils.Compute_Sizes import get_feat_size_lightning
from Models.Feature_Extractor import Feature_Extraction_Layer
from Models.Lightning_Models import (
    AST,
    Custom_TIMM_Model,
    Histogram_Lightning,
    LinearNetwork,
    NEHD_Lightning,
    PANN_Lightning,
)
from Models.NEHD import NEHDLayer


def initialize_models(data_module, network_parameters, filename):

    audio_feature_extractor = Feature_Extraction_Layer(
        audio_feature=network_parameters["audio_feature"],
        sample_rate=network_parameters["sample_rate"],
        window_length=network_parameters["window_length"],
        hop_length=network_parameters["hop_length"],
        freq_bins=network_parameters["freq_bins"],
        padding=network_parameters["feature_ext_padding"],
    )

    intermediate_feature = initialize_model(
        model_name=network_parameters["intermediate_feature"],
        network_parameters=network_parameters,
        filename=filename,
    )

    network_parameters["num_ftrs"] = get_feat_size_lightning(
        data_module,
        model=intermediate_feature,
        audio_feature_extractor=audio_feature_extractor,
    )

    base_model = initialize_model(
        model_name=network_parameters["base_model"],
        network_parameters=network_parameters,
        filename=filename,
    )
    return audio_feature_extractor, intermediate_feature, base_model


def initialize_model(model_name, network_parameters, filename):
    print(model_name)

    if network_parameters["intermediate_feature"] == None:
        num_input_features = 1
    else:
        num_input_features = network_parameters["out_channels"]

    if model_name == "nehd" or model_name == "ehd":
        model = NEHDLayer(
            network_parameters["in_channels"],
            network_parameters["window_size"],
            mask_size=network_parameters["mask_size"],
            num_bins=network_parameters["num_bins"],
            stride=network_parameters["stride"],
            normalize_count=network_parameters["normalize_count"],
            normalize_bins=network_parameters["normalize_bins"],
            EHD_init=network_parameters["feature_init"],
            learn_no_edge=network_parameters["learn_transform"],
            learn_kernel=network_parameters["learn_edge_kernels"],
            learn_hist=network_parameters["learn_hist"],
            threshold=network_parameters["threshold"],
            angle_res=network_parameters["angle_res"],
            dilation=network_parameters["dilation"],
            normalize_kernel=network_parameters["normalize_kernel"],
            aggregation_type=network_parameters["aggregation_type"],
            is_ehd=network_parameters["is_ehd"],
        )
        return NEHD_Lightning(
            model,
            network_parameters["num_epochs"],
            network_parameters["in_channels"],
            network_parameters["out_channels"],
            network_parameters["angle_res"],
            network_parameters["normalize_kernel"],
            network_parameters["mask_size"],
            filename,
        )
    elif model_name == "histogram":
        model = HistogramLayer(
            in_channels=network_parameters["in_channels"],
            kernel_size=network_parameters["window_size"],
            num_bins=network_parameters["num_bins"],
            stride=network_parameters["stride"],
            normalize_count=True,
            normalize_bins=True,
        )
        return Histogram_Lightning(
            model=model,
            epochs=network_parameters["num_epochs"],
            num_bins=network_parameters["num_bins"],
            filename=filename,
        )
    elif model_name == "linear":
        return LinearNetwork(
            network_parameters["num_ftrs"], network_parameters["num_classes"]
        )
    elif model_name == "pann":
        return PANN_Lightning(
            num_input_features=num_input_features,
            num_classes=network_parameters["num_classes"],
        )
    elif model_name == "resnet50" or model_name == "vit" or model_name == "convnext":
        return Custom_TIMM_Model(
            num_input_features=num_input_features,
            num_classes=network_parameters["num_classes"],
            model_name=model_name,
        )

    elif model_name == "ast":
        return AST(
            max_length=60,
            num_input_features=num_input_features,
            num_classes=network_parameters["num_classes"],
            final_output="CLS",
        )
