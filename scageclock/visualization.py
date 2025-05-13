
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

## TODO: optimize the plots
def plot_age_prediction_performance(real_age: List[float],
                    predicted_age: List[float],
                    outdir: str,
                    method_name: str = "CatBoost",
                    filename: str = "evaluation_plot.jpg",
                    text_x: int = 10,
                    text_y: int = 60,
                    figsize: tuple[int, int] = (10, 6),
                    level_type: str = "cell",
                    xlabel: str = "cell donor age",
                    ylabel: str = "predicted cell donor age"):
    """
    plot the correlation between true age and predicted age at either cell level or individual-donor level
    :param real_age:  List of true age values
    :param predicted_age: List of predicted age values
    :param outdir: figure output directory
    :param method_name: model method name
    :param filename: output file name
    :param text_x: r-squared label x location
    :param text_y: r-sequared label y location
    :param figsize: figure size
    :param level_type: "cell" or "donor" level age
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :return:
    """
    if not level_type in ["cell","donor"]:
        print("only cell or donor level_type is supported")
        return False

    x = np.array(real_age)
    y = np.array(predicted_age)

    # Create a linear regression model
    model2 = LinearRegression()
    model2.fit(x.reshape(-1, 1), y)

    # Calculate R-squared
    r_squared = model2.score(x.reshape(-1, 1), y)

    plt.figure(figsize=figsize)

    # Create scatter plot
    plt.scatter(x, y)

    # Plot the regression line
    plt.plot(x, model2.predict(x.reshape(-1, 1)), color='red')

    # Add R-squared value to the plot
    plt.text(text_x, text_y, f'R-squared = {r_squared:.2f}', fontsize=12)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{method_name} Aging Clock Performace ({level_type} level)")
    plt.savefig(f"{outdir}/{filename}")
    plt.close()
    return True

def plot_age_prediction_performance_simple(real_age: List[float],
                    predicted_age: List[float],
                    method_name: str = "scageclock",
                    text_x: int = 10,
                    text_y: int = 60,
                    figsize: tuple[int, int] = (10, 6),
                    level_type: str = "cell",
                    xlabel: str = "cell age",
                    ylabel: str = "predicted cell age"):
    """
    plot the correlation between true age and predicted age at either cell level or individual-donor level
    :param real_age:  List of true age values
    :param predicted_age: List of predicted age values
    :param method_name: model method name
    :param text_x: r-squared label x location
    :param text_y: r-sequared label y location
    :param figsize: figure size
    :param level_type: "cell" or "donor" level age
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :return:
    """
    if not level_type in ["cell","donor"]:
        print("only cell or donor level_type is supported")
        return False

    x = np.array(real_age)
    y = np.array(predicted_age)

    # Create a linear regression model
    model2 = LinearRegression()
    model2.fit(x.reshape(-1, 1), y)

    # Calculate R-squared
    r_squared = model2.score(x.reshape(-1, 1), y)

    plt.figure(figsize=figsize)

    # Create scatter plot
    plt.scatter(x, y)

    # Plot the regression line
    plt.plot(x, model2.predict(x.reshape(-1, 1)), color='red')

    # Add R-squared value to the plot
    plt.text(text_x, text_y, f'R-squared = {r_squared:.2f}', fontsize=12)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{method_name} Aging Clock Performace ({level_type} level)")
    plt.show()
    return True

def plot_age_prediction_performance_simple_diagonal_line(real_age: List[float],
                    predicted_age: List[float],
                    figsize: tuple[int, int] = (10, 6),
                    level_type: str = "cell",
                    xlabel: str = "cell age",
                    ylabel: str = "predicted cell age"):


    # Convert lists to numpy arrays for easier manipulation
    true_ages = np.array(real_age)
    predicted_ages = np.array(predicted_age)

    # Calculate the coefficients for the linear regression line
    coefficients = np.polyfit(true_ages, predicted_ages, 1)
    polynomial = np.poly1d(coefficients)

    # Create a scatter plot
    plt.figure(figsize=figsize)  # Optional: Set figure size for better visualization
    plt.scatter(true_ages, predicted_ages, color='blue', alpha=0.6, edgecolors='w', s=100)

    # Add a diagonal line for reference (y = x)
    plt.plot([min(true_ages + predicted_ages), max(true_ages + predicted_ages)],
             [min(true_ages + predicted_ages), max(true_ages + predicted_ages)],
             color='red', linestyle='--', linewidth=2, label='y = x')

    # Plot the linear regression line
    x_values = np.linspace(min(true_ages), max(true_ages), 100)  # Generate x values for the regression line
    y_values = polynomial(x_values)  # Calculate y values using the polynomial
    plt.plot(x_values, y_values, color='red', linewidth=2, label='Linear Fit')

    # Set plot limits and scale
    min_age = min(min(true_ages), min(predicted_ages))
    max_age = max(max(true_ages), max(predicted_ages))
    plt.xlim(0, max_age + 5)  # Adding a bit of padding
    plt.ylim(0, max_age + 5)

    # Set equal scaling for both axes
    plt.gca().set_aspect('equal', adjustable='box')

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"True vs Predicted Age ({level_type})")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def age_diff_density_plot(true_age, predicted_age):
    data = list(predicted_age- true_age)

    # Using seaborn to create a density plot
    sns.set(style="whitegrid")  # Optional: set a style for the plot
    plt.figure(figsize=(8, 6))  # Optional: set the figure size

    sns.kdeplot(data, fill=True, color="blue", alpha=0.5)  # kdeplot for density plot

    # Adding labels and title
    plt.title('Density Plot of Age-difference (Predicted Age - True Age)')
    plt.xlabel('Age Difference')
    plt.ylabel('Density')

    # Show the plot
    plt.show()

def plot_catboost_eval(train_metrics_df,
                       figsize: tuple[int, int] = (12,6),
                       x: str = "step",
                       y: str = "RMSE",
                       label: str = "label",
                       save_path: str = None):
    sns.set_theme(rc={'figure.figsize': figsize})
    sns.scatterplot(data=train_metrics_df, x=x, y=y, hue=label)

    if save_path:
        plt.savefig(save_path)

def barplot(df,
            x_col,
            y_col,
            title,
            xlabel,
            ylabel,
            savefile: str | None = None,
            ylim = None,
            figsize = (6,5)):
    # Plot the correlations
    plt.figure(figsize=figsize)

    if ylim is None:
        ylim_min = 0
        ylim_max = max(df[y_col])*1.1
        ylim = (ylim_min, ylim_max)

    # Subplot 1: Correlation
    sns.barplot(x=x_col, y=y_col, data=df, palette='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim[0], ylim[1])
    # Improve x-axis label visibility
    plt.xticks(
        rotation=45,
        ha='right',  # Align right for better readability
        fontsize=9  # Slightly smaller font if needed
    )

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()
