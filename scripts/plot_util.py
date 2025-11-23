import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PlotHelper:
    """
    A helper class for creating various types of plots using matplotlib and seaborn.
    Follows PEP8 conventions.
    """

    def __init__(self, figsize=(10, 6), style="seaborn-v0_8"):
        """
        Initialize the PlotHelper.

        Args:
            figsize (tuple): Default figure size for plots.
            style (str): Matplotlib style to use.
        """ 
        plt.style.use(style)
        self.figsize = figsize

    def histogram(self, data, column=None, bins=30, **kwargs):
        """
        Create a histogram.

        Args:
            data (pd.DataFrame or array-like): Data to plot.
            column (str, optional): Column name if data is DataFrame.
            bins (int): Number of bins.
            **kwargs: Additional arguments for plt.hist or sns.histplot.
        """
        plt.figure(figsize=self.figsize)
        if isinstance(data, pd.DataFrame) and column:
            sns.histplot(data[column], bins=bins, **kwargs)
        else:
            plt.hist(data, bins=bins, **kwargs)
        plt.show()

    def scatter(self, x, y, data=None, **kwargs):
        """
        Create a scatter plot.

        Args:
            x (str or array-like): X data or column name.
            y (str or array-like): Y data or column name.
            data (pd.DataFrame, optional): DataFrame if x and y are column names.
            **kwargs: Additional arguments for plt.scatter or sns.scatterplot.
        """
        plt.figure(figsize=self.figsize)
        if data is not None:
            sns.scatterplot(x=x, y=y, data=data, **kwargs)
        else:
            plt.scatter(x, y, **kwargs)
        plt.show()

    def line(self, x, y, data=None, **kwargs):
        """
        Create a line plot.

        Args:
            x (str or array-like): X data or column name.
            y (str or array-like): Y data or column name.
            data (pd.DataFrame, optional): DataFrame if x and y are column names.
            **kwargs: Additional arguments for plt.plot or sns.lineplot.
        """
        plt.figure(figsize=self.figsize)
        if data is not None:
            sns.lineplot(x=x, y=y, data=data, **kwargs)
        else:
            plt.plot(x, y, **kwargs)
        plt.show()

    def bar(self, x, height, data=None, **kwargs):
        """
        Create a bar plot.

        Args:
            x (str or array-like): X data or column name.
            height (str or array-like): Height data or column name.
            data (pd.DataFrame, optional): DataFrame if x and height are column names.
            **kwargs: Additional arguments for plt.bar or sns.barplot.
        """
        plt.figure(figsize=self.figsize)
        if data is not None:
            sns.barplot(x=x, y=height, data=data, **kwargs)
        else:
            plt.bar(x, height, **kwargs)
        plt.show()

    def boxplot(self, data, x=None, y=None, **kwargs):
        """
        Create a box plot.

        Args:
            data (pd.DataFrame or array-like): Data to plot.
            x (str, optional): X column name.
            y (str, optional): Y column name.
            **kwargs: Additional arguments for sns.boxplot.
        """
        plt.figure(figsize=self.figsize)
        if isinstance(data, pd.DataFrame):
            sns.boxplot(data=data, x=x, y=y, **kwargs)
        else:
            sns.boxplot(data=data, **kwargs)
        plt.show()

    def heatmap(self, data, **kwargs):
        """
        Create a heatmap.

        Args:
            data (pd.DataFrame or array-like): Data to plot.
            **kwargs: Additional arguments for sns.heatmap.
        """
        plt.figure(figsize=self.figsize)
        sns.heatmap(data, **kwargs)
        plt.show()

    def plot_correlation_matrix(self, data, **kwargs):
        """
        Create a correlation matrix heatmap.

        Args:
            data (pd.DataFrame): DataFrame for which to plot the correlation matrix.
            **kwargs: Additional arguments for sns.heatmap.
        """
        corr = data.corr()
        plt.figure(figsize=self.figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', **kwargs)
        plt.show()

    def plot_predictions(self, y_true, y_pred, **kwargs):
        """
        Plot true vs predicted values.

        Args:
            y_true (array-like): True values.
            y_pred (array-like): Predicted values.
            **kwargs: Additional arguments for plt.scatter.
        """
        plt.figure(figsize=self.figsize)
        plt.scatter(y_true, y_pred, **kwargs)
        plt.plot([min(y_true), max(y_true)], [min(y_true),
                 max(y_true)], color='red', linestyle='--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.show()

    def grid_view(self, plots, nrows, ncols, figsize=None):
        """
        Create a grid of subplots.

        Args:
            plots (list): List of plot functions or data to plot.
            nrows (int): Number of rows in the grid.
            ncols (int): Number of columns in the grid.
            figsize (tuple, optional): Figure size for the grid.
        """
        if figsize is None:
            figsize = (ncols * 5, nrows * 4)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if nrows * ncols > 1 else [axes]

        for i, plot_func in enumerate(plots):
            if i < len(axes):
                plt.sca(axes[i])
                plot_func()
        plt.tight_layout()
        plt.show()



x = PlotHelper()
print(x)