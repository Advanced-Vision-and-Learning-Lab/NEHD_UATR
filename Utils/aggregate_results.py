# -*- coding: utf-8 -*-
"""
Created on Thursday April 25 22:32:00 2024
Aggregate results from saved models
@author: salimalkharsa
"""
import os
import glob
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import json


def aggregate_tensorboard_logs(root_dir):
    aggregated_results = defaultdict(lambda: defaultdict(list))

    # Traverse through the directory structure
    for run_dir in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        # Look for event files within each run directory
        event_file = os.path.join(
            run_path, "lightning_logs", "version_0", "events.out.tfevents.*"
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

    return final_aggregated_results


if __name__ == "__main__":
    root_dir = "Saved_Models/Fine_Tuning/UCMerced/NLBP"
    results = aggregate_tensorboard_logs(root_dir)

    # Output aggregated results to JSON
    with open("aggregated_results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Aggregated results saved to 'aggregated_results.json'.")
