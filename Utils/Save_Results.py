# -*- coding: utf-8 -*-
"""
Created on Thursday April 25 22:32:00 2024
Modified on  Fri Jan 10 19:15:42 2025

Save aggregate results from saved models
@author: salimalkharsa, jpeeples, aagashe
"""
# Python standard libraries
from __future__ import print_function
from __future__ import division
import os
import glob
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import json
import pdb


def generate_filename(Network_parameters, split=0):
    # Generate filename for saving results
    # Baseline model
    filename = "{}/{}/{}/{}/Run_{}/".format(
        Network_parameters["folder"],
        Network_parameters["base_model"],
        Network_parameters["intermediate_feature"],
        Network_parameters["audio_feature"],
        split + 1,
    )

    # Create directory if it does not exist
    if not os.path.exists(filename):
        os.makedirs(filename)
    return filename


def plot_avg_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    show_percent=True,
    ax=None,
    fontsize=12,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute average CM values//
    std_cm = np.int64(np.ceil(np.std(cm, axis=2)))
    cm = np.int64(np.ceil(np.mean(cm, axis=2)))

    # Generate new figure if needed
    if ax is None:
        fig, ax = plt.subplots()

    if show_percent:
        cm_percent = 100 * cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_percent_std = (
            100 * std_cm.astype("float") / (std_cm.sum(axis=1)[:, np.newaxis] + 10e-6)
        )
        im = ax.imshow(cm_percent, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)
        plt.title(title)
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize)

    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=fontsize)
    plt.yticks(tick_marks, classes, fontsize=fontsize)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if show_percent:
            s = str(
                format(cm[i, j], fmt)
                + "±"
                + format(std_cm[i, j], fmt)
                + "\n"
                + "("
                + format(cm_percent[i, j], ".2f")
                + "±"
                + "\n"
                + format(cm_percent_std[i, j], ".2f")
                + "%)"
            )
        else:
            s = str(format(cm[i, j], fmt) + "±" + format(std_cm[i, j], fmt))

        ax.text(
            j,
            i,
            s,
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=fontsize // 1.3,
        )

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
    )
    # ylabel="True label",
    # xlabel="Predicted label")

    ax.set_ylim((len(classes) - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=45)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.tight_layout()


def aggregate_tensorboard_logs(root_dir, save_dir, dataset):
    aggregated_results = defaultdict(lambda: defaultdict(list))

    # Create save directory if it doesn't exist
    save_dir = "{}/{}".format(root_dir, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Traverse through the directory structure
    for run_dir in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        # Look for event files within each run directory
        event_file = os.path.join(
            run_path, "lightning_logs", "Val_Test", "events.out.tfevents.*"
        )
        event_files = glob.glob(event_file)

        for event_file in event_files:
            event_acc = EventAccumulator(event_file)
            event_acc.Reload()

            # Extract scalar data from event file
            tags = event_acc.Tags()["scalars"]
            for tag in tags:
                if any(phase in tag for phase in ["train", "val", "test"]):
                    phase, metric = tag.split("_", 1)
                    events = event_acc.Scalars(tag)
                    values = [event.value for event in events]
                    aggregated_results[metric][phase].extend(values)

    # Aggregate metrics
    final_aggregated_results = {}
    for metric, phases in aggregated_results.items():
        phase_results = {}
        for phase, values in phases.items():
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            phase_results[phase] = {"mean": mean, "std": std}
        final_aggregated_results[metric] = phase_results

    # Save results in JSON file
    with open(
        "{}/{}_aggregated_metrics.json".format(save_dir, dataset), "w"
    ) as outfile:
        json.dump(final_aggregated_results, outfile)

    return final_aggregated_results


def aggregate_and_visualize_confusion_matrices(
    root_dir,
    save_dir,
    dataset,
    label_names=None,
    cmap="Blues",
    threshold=10,
    figsize=(15, 10),
    fontsize=24,
    title=False,
):
    # Create save directory if it doesn't exist
    save_dir = "{}/{}".format(root_dir, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_aggregated_matrix = None
    aggregated_matrix_list = []
    test_matrix_count = 0

    # Aggregate confusion matrices
    for run_dir in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        # Look for CSV files containing confusion matrices
        csv_files = [
            f
            for f in os.listdir(run_path)
            if f.endswith(".csv") and "confusion_matrix" in f
        ]
        for csv_file in csv_files:
            # Read the test matrix
            if "test" in csv_file:
                matrix = pd.read_csv(
                    os.path.join(run_path, csv_file), index_col=0
                ).to_numpy()
                if test_aggregated_matrix is None:
                    # test_aggregated_matrix = matrix
                    aggregated_matrix_list.append(matrix[..., np.newaxis])
                else:
                    # test_aggregated_matrix += matrix
                    aggregated_matrix_list.append(matrix[..., np.newaxis])
                test_matrix_count += 1

    # Convert list to numpy array
    aggregated_matrix_list = np.concatenate(aggregated_matrix_list, axis=-1)
    test_mean_matrix = np.mean(aggregated_matrix_list, axis=-1)

    # Generate heatmap

    # Cases: 1) Too many classes, 2) Few classes
    # If the number of classes is greater than 5, then the heatmap will not display the class names or the values in the boxes

    if title:
        title_name = "{} Confusion Matrix".format(dataset)
    else:
        title_name = None

    if label_names is not None and len(label_names) <= threshold:
        plt.figure(figsize=figsize)
        fig, ax = plt.subplots(figsize=figsize)
        plot_avg_confusion_matrix(aggregated_matrix_list, label_names, title=title_name)

        # Save the heatmap as an image
        plt.savefig(
            os.path.join(
                save_dir, "{}_aggregated_confusion_matrix.png".format(dataset)
            ),
            bbox_inches="tight",
        )
        plt.close()
        return save_dir
    else:
        plt.figure(figsize=figsize)
        sns.heatmap(test_mean_matrix, cmap=cmap, cbar=True, annot=False)
        plt.title(title_name, fontsize=fontsize)
        plt.xlabel("Predicted Label", fontsize=fontsize - 4)
        plt.ylabel("True Label", fontsize=fontsize - 4)
        # Save the heatmap as an image
        plt.savefig(
            os.path.join(save_dir, "aggregated_confusion_matrix_" + dataset + ".png"),
            bbox_inches="tight",
        )
        plt.close()
        return save_dir
