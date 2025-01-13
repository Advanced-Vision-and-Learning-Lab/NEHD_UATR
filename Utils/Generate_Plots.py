# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 20:49:45 2023
Function to plot results during training, validation and testing
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

## Local external libraries
from Utils.Save_Results import generate_filename


def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def plot_histogram(centers, widths, epoch, phase, angle_res, filename):

    # Plot some images
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.subplots_adjust(right=0.75)
    angles = np.arange(0, 360, angle_res)

    num_bins = len(centers)
    handles = []
    for temp_ang in range(0, num_bins):

        toy_data = np.linspace(
            centers[temp_ang] - 6 * abs(widths[temp_ang]),
            centers[temp_ang] + 6 * abs(widths[temp_ang]),
            300,
        )
        y = scipy.stats.norm.pdf(toy_data, centers[temp_ang], abs(widths[temp_ang]))
        y = y / y.max()
        plt.plot(toy_data, y)
        handles.append(
            "Bin {}: \u03BC = {:.2f}, \u03C3 = {:.2f}".format(
                temp_ang + 1, centers[temp_ang], widths[temp_ang]
            )
        )
    plt.legend(handles, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.suptitle(
        ("{}-Bin Histogram for {} " + "Feature Maps Epoch {}").format(
            num_bins, len(angles), epoch + 1
        )
    )

    filename = filename + phase + "/" + "Histograms/"

    if not os.path.exists(filename):
        os.makedirs(filename)

    try:
        if epoch is not None:
            plt.suptitle("Epoch {} during {} phase".format(epoch + 1, phase))
            plt.savefig(
                filename + "Epoch_{}_Phase_{}.png".format(epoch + 1, phase), dpi=fig.dpi
            )
        else:
            plt.suptitle("Best Epoch for {} phase".format(phase))
            plt.savefig(filename + "Best_Epoch_Phase_{}.png".format(phase), dpi=fig.dpi)
    except:
        pass
    plt.close(fig=fig)


def plot_kernels(EHD_masks, Hist_masks, phase, epoch, in_channels, angle_res, filename):

    # Plot some images
    fig, ax = plt.subplots(3, Hist_masks.size(0), figsize=(24, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    angles = np.arange(0, 360, angle_res)
    plot_min = np.amin(EHD_masks)
    if np.amin(Hist_masks.detach().cpu().numpy()) < plot_min:
        plot_min = np.amin(Hist_masks.detach().cpu().numpy())
    plot_max = np.amax(EHD_masks)
    if np.amax(Hist_masks.detach().cpu().numpy()) > plot_max:
        plot_max = np.amax(Hist_masks.detach().cpu().numpy())

    # Remove extra dimension on histogram masks tensor
    Hist_masks = Hist_masks.squeeze(1)
    num_orientations = Hist_masks.size(0) // in_channels

    # ^^ Ax size is 8 when running debug (previously had -1)
    for temp_ang in range(0, num_orientations):

        if temp_ang == (num_orientations - 1):
            ax[0, temp_ang].set_title("No Edge (N/A)")
            ax[1, temp_ang].set_title("Bin {}".format(temp_ang + 1))
            ax[2, temp_ang].set_title("No Edge")
            ax[0, temp_ang].set_yticks([])
            ax[1, temp_ang].set_yticks([])
            ax[2, temp_ang].set_yticks([])
            ax[0, temp_ang].set_xticks([])
            ax[1, temp_ang].set_xticks([])
            ax[2, temp_ang].set_xticks([])
            im = ax[0, temp_ang].imshow(np.zeros(EHD_masks[temp_ang - 1].shape))
            im2 = ax[1, temp_ang].imshow(Hist_masks[temp_ang].detach().cpu().numpy())
            im3 = ax[2, temp_ang].imshow(
                abs(
                    Hist_masks[temp_ang].detach().cpu().numpy()
                    - np.zeros(EHD_masks[temp_ang - 1].shape)
                )
            )
        else:
            ax[0, temp_ang].set_title(str(angles[temp_ang]) + "\N{DEGREE SIGN}")
            ax[1, temp_ang].set_title("Bin {}".format(temp_ang + 1))
            ax[2, temp_ang].set_title(str(angles[temp_ang]) + "\N{DEGREE SIGN}")
            ax[0, temp_ang].set_yticks([])
            ax[1, temp_ang].set_yticks([])
            ax[2, temp_ang].set_yticks([])
            ax[0, temp_ang].set_xticks([])
            ax[1, temp_ang].set_xticks([])
            ax[2, temp_ang].set_xticks([])
            im = ax[0, temp_ang].imshow(EHD_masks[temp_ang])
            im2 = ax[1, temp_ang].imshow(Hist_masks[temp_ang].detach().cpu().numpy())
            diff = abs(
                Hist_masks[temp_ang].detach().cpu().numpy() - EHD_masks[temp_ang]
            )
            im3 = ax[2, temp_ang].imshow(diff)

        plt.colorbar(im, ax=ax[0, temp_ang], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax[1, temp_ang], fraction=0.046, pad=0.04)
        plt.colorbar(im3, ax=ax[2, temp_ang], fraction=0.046, pad=0.04)

    ax[0, 0].set_ylabel("EHD True Masks", rotation=90, size="small")
    ax[1, 0].set_ylabel("Hist Layer Masks", rotation=90, size="small")
    ax[2, 0].set_ylabel("Absolute Difference", rotation=90, size="small")
    plt.tight_layout()

    filename = filename + phase + "/" + "Masks/"

    if not os.path.exists(filename):
        os.makedirs(filename)

    try:
        if epoch is not None:
            plt.suptitle("Epoch {} during {} phase".format(epoch + 1, phase))
            plt.savefig(
                filename + "Epoch_{}_Phase_{}.png".format(epoch + 1, phase), dpi=fig.dpi
            )
        else:
            plt.suptitle("Best Epoch for {} phase".format(phase))
            plt.savefig(filename + "Best_Epoch_Phase_{}.png".format(phase), dpi=fig.dpi)
    except:
        pass
    plt.close(fig=fig)
