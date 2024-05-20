import numpy as np
import flygym as flygym
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import imageio
import cv2


def plot_chasing(time, fly0_speeds, fly1_speeds, proximities, smooth=True):
    """
    Plot the speed of the chasing and target flies, as well as the proximity between them, during simulation.

    Parameters
    ----------
    time : numpy array
        The time points at which the data was recorded.
    fly0_speeds : numpy array
        The speed of the chasing fly at each time point.
    fly1_speeds : numpy array
        The speed of the target fly at each time point.
    proximities : numpy array
        The proximity between the flies at each time point.
    smooth : bool
        Whether to smooth the signals using a moving average filter.
    """
    # Smooth the signals
    if smooth:
        window_size = 1000  # Define the size of the moving average window
        window = np.ones(window_size) / window_size  # Create a uniform window

        fly0_speeds = np.convolve(fly0_speeds, window, mode='same')  # Apply the moving average filter
        fly1_speeds = np.convolve(fly1_speeds, window, mode='same')
        proximities = np.convolve(proximities, window, mode='same')

    # Create a color palette
    palette = sns.color_palette("Set2", 3)

    fig, ax1 = plt.subplots(figsize=(20, 6))
    line1 = ax1.plot(time, fly0_speeds, color=palette[0], label="Chasing fly speed")
    line2 = ax1.plot(time, fly1_speeds, color=palette[1], label="Target fly speed")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("speed (m/s)")

    ax2 = ax1.twinx()
    line3 = ax2.plot(time, proximities, color=palette[2], label="Proximity of the flies")
    ax2.set_ylabel("Proximity (pixels)")

    # Create a combined legend for all lines
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax2.legend([line3[0]], [line3[0].get_label()], loc='upper right')

    fig.tight_layout()
    plt.show()

def plot_visual_detection(time, left_input, right_input, proximities):
    """
    Plot the object detection signals from the left and right eyes, as well as the proximity between the flies, during simulation.

    Parameters
    ----------
    time : numpy array
        The time points at which the data was recorded.
    left_input : numpy array
        The object detection signals from the left eye at each time point.
    right_input : numpy array
        The object detection signals from the left and right eyes at each time point.
    proximities : numpy array
        The proximity between the flies at each time point.
    """
    # Create a color palette
    palette = sns.color_palette("Set2", 3)

    fig, ax1 = plt.subplots(figsize=(20, 6))
    line1 = ax1.plot(time, left_input, color=palette[0], label="Left object detection")
    line2 = ax1.plot(time, right_input, color=palette[1], label="right object detection")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("Object detection")

    ax2 = ax1.twinx()
    line3 = ax2.plot(time, proximities, color=palette[2], label="Wing extension")
    ax2.set_ylabel("Wing extension state")

    # Create a combined legend for all lines
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax2.legend([line3[0]], [line3[0].get_label()], loc='upper right')

    fig.tight_layout()
    plt.show()

def plot_overlayed_frames(birdeye_cam_frames, save=False):
    """
    Plot the overlayed frames of the birdeye camera.

    Parameters
    ----------
    birdeye_cam_frames : numpy array
        The frames of the birdeye camera.
    save : bool
        Whether to save the plot as an image.
    """
    frame_indices = np.arange(0, len(birdeye_cam_frames), 30)[:8]
    snapshots = [birdeye_cam_frames[i] for i in frame_indices]
    background = np.median(snapshots, axis=0).astype("uint8")

    imgs = []

    for i, img in enumerate(snapshots):
        is_background = np.isclose(img, background, atol=1).all(axis=2)
        img_alpha = np.ones((img.shape[0], img.shape[1], 4)) * 255
        img_alpha[:, :, :3] = img
        img_alpha[is_background, 3] = 0
        img_alpha = img_alpha.astype(np.uint8)
        imgs.append(img_alpha.copy())

    dpi = 72
    h, w = background.shape[:2]

    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=72)
    ax.imshow(background)
    ax.axis("off")

    for i, img in enumerate(imgs):
        ax.imshow(img, alpha=(i + 1) / len(imgs))

    fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    if save:
        plt.savefig("outputs/overlayed_frames.png", dpi=dpi)
    else:
        plt.show()