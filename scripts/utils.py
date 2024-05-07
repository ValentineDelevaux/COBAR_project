import numpy as np
import flygym as flygym
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import imageio
import cv2
from flygym.vision import add_insets

def adjust_camera_to_include_flies(cam, positions_fly0, positions_fly1):
    # Calculate the midpoint between the two flies
    midpoint = (flies[0].position + flies[1].position) / 2

    # Set the camera's position to the midpoint
    cam.set_position(midpoint)

    # Orient the camera to face the midpoint
    # cam.set_orientation(midpoint)


def save_video_with_vision_insets(
    sim, cam, path, visual_input_hist, positions_fly0, positions_fly1, stabilization_time=0.02
):
    """Save a list of frames as a video with insets showing the visual
    experience of the fly. This is almost a drop-in replacement of
    ``NeuroMechFly.save_video`` but as a static function (instead of a
    class method) and with an extra argument ``visual_input_hist``.

    Parameters
    ----------
    sim : Simulation
        The Simulation object.
    cam : Camera
        The Camera object that has been used to generate the frames.
    path : Path
        Path of the output video will be saved. Should end with ".mp4".
    visual_input_hist : List[np.ndarray]
        List of ommatidia readings. Each element is an array of shape
        (2, N, 2) where N is the number of ommatidia per eye.
    stabilization_time : float, optional
        Time (in seconds) to wait before starting to render the video.
        This might be wanted because it takes a few frames for the
        position controller to move the joints to the specified angles
        from the default, all-stretched position. By default 0.02s
    """
    if len(visual_input_hist) != len(cam._frames):
        raise ValueError(
            "Length of `visual_input_hist` must match the number of "
            "frames in the `NeuroMechFly` object. Save the visual input "
            "every time a frame is rendered, i.e. when `.render()` returns "
            "a non-`None` value."
        )

    adjust_camera_to_include_flies(cam, positions_fly0, positions_fly1)

    num_stab_frames = int(np.ceil(stabilization_time / cam._eff_render_interval))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving video to {path}")
    with imageio.get_writer(path, fps=cam.fps) as writer:
        for i, (frame, visual_input) in enumerate(zip(cam._frames, visual_input_hist)):
            if i < num_stab_frames:
                continue
            frame = add_insets(sim.flies[0].retina, frame, visual_input)
            writer.append_data(frame)

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