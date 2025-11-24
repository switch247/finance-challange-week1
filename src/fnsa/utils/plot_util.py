import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- FIX: Robust Style Setting with Fallback ---
# This block attempts to set a nice style, but falls back to 'default' if
# style sheet files cannot be found due to environment/import path issues.
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('default')
    print("Warning: Could not load 'seaborn-v0_8' style. Falling back to 'default'.")


class PlotHelper:
    """
    A helper class for creating various types of plots using matplotlib and seaborn.
    Follows PEP8 conventions and ensures figure management.
    """

    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize

    def _setup_figure(self, title=None):
        """Internal method to create a figure and set the title."""
        plt.figure(figsize=self.figsize)
        if title:
            plt.title(title, fontsize=16)

    def histogram(self, data, column=None, bins=30, **kwargs):
        """
        Create a histogram (Distribution of data).

        Args:
            data (pd.DataFrame or array-like): Data to plot.
            column (str, optional): Column name if data is DataFrame.
            bins (int): Number of bins.
            **kwargs: Additional arguments for sns.histplot.
        """
        title = kwargs.pop('title', 'Data Distribution')
        self._setup_figure(title=title)

        plot_data = data[column] if isinstance(
            data, pd.DataFrame) and column else data

        sns.histplot(plot_data, bins=bins, kde=True, **kwargs)
        plt.xlabel(column or 'Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', alpha=0.5)
        plt.show()

    def scatter(self, data, x, y, **kwargs):
        """
        Create a scatter plot (Relationship between two variables).

        Args:
            data (pd.DataFrame): DataFrame with x and y columns.
            x (str): Column name for x.
            y (str): Column name for y.
            **kwargs: Additional arguments for sns.scatterplot.
        """
        title = kwargs.pop('title', f'Scatter Plot: {x} vs {y}')
        self._setup_figure(title=title)
        sns.scatterplot(x=x, y=y, data=data, **kwargs)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def line(self, data, x, y, **kwargs):
        """
        Create a line plot (Trend over Time/Sequence). Crucial for time series.

        Args:
            data (pd.DataFrame): DataFrame with x and y columns.
            x (str): Column name for x.
            y (str or int): Column name for y.
            **kwargs: Additional arguments for sns.lineplot.
        """
        title = kwargs.pop('title', f'Line Plot: {y} over {x}')
        self._setup_figure(title=title)

        # Check if x is the index (common in time series plots from resample().reset_index())
        if data.index.name == x or x == data.index.name:
            plot_data = data.reset_index()
            sns.lineplot(x=x, y=y, data=plot_data, **kwargs)
        else:
            sns.lineplot(x=x, y=y, data=data, **kwargs)

        plt.xlabel(x.capitalize(), fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def line_plot(self, data, x, y, **kwargs):
        """Alias for line method to match common usage."""
        return self.line(data, x, y, **kwargs)

    def bar(self, data, x, y, **kwargs):
        """
        Create a bar plot (Comparison of categorical groups).

        Args:
            data (pd.DataFrame): DataFrame with x (category) and y (value) columns.
            x (str): Column name for x (categories).
            y (str or int): Column name for y (values).
            **kwargs: Additional arguments for sns.barplot.
        """
        title = kwargs.pop('title', f'Bar Plot: {y} by {x}')
        self._setup_figure(title=title)

        # Use sns.barplot
        sns.barplot(x=x, y=y, data=data, **kwargs)

        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        # Improve readability for categories
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def heatmap(self, data, **kwargs):
        """
        Create a heatmap (Correlation or matrix visualization).
        """
        title = kwargs.pop('title', 'Heatmap Visualization')
        self._setup_figure(figsize=kwargs.pop('figsize', (10, 8)), title=title)

        plot_data = data.select_dtypes(include=np.number).corr(
        ) if isinstance(data, pd.DataFrame) else data

        sns.heatmap(
            plot_data,
            annot=kwargs.pop('annot', True),
            fmt=kwargs.pop('fmt', ".2f"),
            cmap=kwargs.pop('cmap', 'coolwarm'),
            linewidths=kwargs.pop('linewidths', 0.5),
            **kwargs
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, data, **kwargs):
        """
        Create a correlation matrix heatmap.
        """
        kwargs['title'] = kwargs.pop('title', 'Correlation Matrix Heatmap')
        return self.heatmap(data, **kwargs)

    def plot_predictions(self, y_true, y_pred, **kwargs):
        """
        Plot true vs predicted values.
        """
        title = kwargs.pop('title', 'True vs Predicted Values')
        self._setup_figure(title=title)

        plt.scatter(y_true, y_pred, alpha=0.7,
                    edgecolors='w', linewidths=0.5, **kwargs)

        # Plot the ideal line (y=x)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color='red',
                 linestyle='--', label='Ideal Prediction')

        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
