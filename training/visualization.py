import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interpn


def get_density_scatter_plot_visualization(
        process_variables=None,
        ignore_value=-9999,
        min_value=0,
        max_value=60,
):
    """
    Creates a density scatter plot visualization for the regression model
    :param process_variables: A function that takes in inputs, labels, and outputs and returns the processed versions
    :return: A function that takes in inputs, labels, and outputs and returns a boxplot visualization
    """

    def density_scatter_visualization(
            inputs, labels, outputs, bins=30, height_range=range(1, max_value), **kwargs
    ):
        inputs = inputs.detach().cpu().numpy().squeeze()
        labels = labels.detach().cpu().numpy().squeeze()
        outputs = outputs.detach().cpu().numpy().squeeze()

        if process_variables is not None:
            inputs, labels, outputs = process_variables(inputs, labels, outputs)

        outputs = outputs[labels != ignore_value]
        labels = labels[labels != ignore_value]

        ax = plt.gca()
        x = np.array(labels).flatten()
        y = np.array(outputs).flatten()

        fig, ax = plt.subplots()

        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = interpn(
            (x_e[:-1], y_e[:-1]),
            data,
            np.vstack([x, y]).T,
            method="splinef2d",
            bounds_error=False,
        )

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        if np.min(z) < 0:
            z = z + abs(np.min(z) + 0.1)
        norm = Normalize(vmin=np.min(z), vmax=np.max(z), clip=True)

        ax.scatter(x, y, c=z, s=0.1, cmap="viridis", norm=norm, **kwargs)
        plt.xlim([min_value, max_value])
        plt.ylim([min_value, max_value])

        fig = plt.gcf()
        plt.grid(False)
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, **kwargs
        )
        cbar.ax.set_ylabel("Density")

        ax.plot(
            height_range, height_range, c="r", linewidth=3, label="x=y", linestyle="--"
        )
        ax.set_xlabel("Ground truth height")
        ax.set_ylabel("Predicted height")
        ax.legend()

        return fig

    return density_scatter_visualization


def get_input_output_visualization(
        process_variables=None,
        transparent_value=None,
        rgb_channels=[2, 1, 0],
):
    """
    Get a visualization function that plots the input and output of the model.
    :param process_variables: A function that processes the variables before plotting.
    :param transparent_value: The value that should be transparent in the output.
    :param rgb_channels: The channels that should be used for the RGB image.
    :return: A function that can be used for visualization.
    """

    def input_output_visualization(inputs, labels, outputs):
        inputs = inputs.cpu().detach().numpy()
        if labels is not None:
            labels = labels.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        if transparent_value is not None:
            outputs[outputs == transparent_value] = None

        if process_variables is not None:
            inputs, labels, outputs = process_variables(inputs, labels, outputs)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        inputs_normalized = np.clip(inputs / 3000, 0, 1)

        # loop over the first two images
        for i in range(2):
            # plot the RGB image in the first column
            axs[i, 0].imshow(
                inputs_normalized[i, rgb_channels, :, :].transpose(1, 2, 0)
            )  # note that we reverse the order to RGB
            axs[i, 0].set_title(f"Image {i + 1}")

            # plot the model output in the second column
            im = axs[i, 1].imshow(outputs[i, 0, :, :], cmap="viridis", vmin=0, vmax=35)
            axs[i, 1].set_title(f"Output {i + 1}")

        # Create colorbar
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        return fig

    return input_output_visualization


def get_visualization_boxplots(
        process_variables=None,
        ignore_value=-9999,
        min_value=0,
        max_value=60,
        step_size=5,
):
    """
    Creates a boxplot visualization for the regression model
    :param process_variables: A function that takes in inputs, labels, and outputs and returns the processed versions
    :return: A function that takes in inputs, labels, and outputs and returns a boxplot visualization
    """

    def visualization_boxplots(inputs, labels, outputs):
        inputs = inputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        if process_variables is not None:
            inputs, labels, outputs = process_variables(inputs, labels, outputs)

        # Reshape labels and outputs into 1D arrays
        labels_1d = labels.flatten()
        outputs_1d = outputs.flatten()

        # Calculate errors
        errors = outputs_1d - labels_1d

        # Only keep non-zero label locations
        non_zero_label_locs = labels_1d != ignore_value
        labels_1d = labels_1d[non_zero_label_locs]
        errors = errors[non_zero_label_locs]

        # Create bins for 'label' array
        bins = np.arange(min_value, max_value, step_size)
        bin_indices = np.digitize(labels_1d, bins)

        # Initialize lists to hold errors for each bin
        bin_errors = [[] for _ in range(len(bins) + 1)]
        bin_counts = [0] * (len(bins) + 1)
        for bin_idx, error in zip(bin_indices, errors):
            bin_errors[bin_idx].append(error)
            bin_counts[bin_idx] += 1

        # Modify x-axis labels with counts
        x_labels = []
        for i in range(len(bin_counts) - 1):
            x_labels.append(f"{bins[i]}-{bins[i] + step_size} (n={bin_counts[i]:,})")
        x_labels.append(f">{bins[-1]} (n={bin_counts[-1]:,})")

        # Create the boxplot
        plt.figure(figsize=(10, 6))
        box_plot = sns.boxplot(data=bin_errors, fliersize=2)

        box_plot.set_xticklabels(x_labels)

        plt.title("Boxplot of Errors for Tree Height Bins")
        plt.xlabel("Label Bins")
        plt.ylabel("Error")
        plt.xticks(rotation=90)  # Make sure that the labels on x-axis don't overlap

        return plt.gcf()  # return the current figure

    return visualization_boxplots
