import numpy as np
import flygym as flygym
import matplotlib.pyplot as plt
import seaborn as sns


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