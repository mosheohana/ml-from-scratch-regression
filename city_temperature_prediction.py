
import matplotlib.pyplot as plt
import pandas as pd
from polynomial_fitting import PolynomialFitting
import numpy as np


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])

    df = df.dropna()

    df = df[(df["Temp"] >= -40) & (df["Temp"] <= 40)]

    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    israel_df = df[df["Country"] == "Israel"]

    # Scatter plot: Temp vs. DayOfYear colored by Year
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        israel_df["DayOfYear"],
        israel_df["Temp"],
        c=israel_df["Year"],
        cmap="tab10",  # Discrete colormap
        alpha=0.7
    )
    plt.colorbar(scatter, label="Year")
    plt.title("Daily Temperature in Israel by Day of Year")
    plt.xlabel("Day of Year")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("israel_temp_scatter.png")
    plt.close()

    # Bar plot: Std. deviation by month
    monthly_std = israel_df.groupby("Month")["Temp"].std()

    plt.figure(figsize=(8, 5))
    monthly_std.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Standard Deviation of Temperature by Month (Israel)")
    plt.xlabel("Month")
    plt.ylabel("Temperature Std. Dev. (°C)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("israel_monthly_std.png")
    plt.close()

    # Question 4 - Exploring differences between countries
    # Group by Country and Month to calculate mean and std of temperature
    grouped = df.groupby(["Country", "Month"])["Temp"].agg(["mean", "std"]).reset_index()

    # Plot: Average temperature by month with error bars, one line per country
    plt.figure(figsize=(10, 6))

    for country in grouped["Country"].unique():
        country_data = grouped[grouped["Country"] == country]
        plt.errorbar(
            country_data["Month"],
            country_data["mean"],
            yerr=country_data["std"],
            label=country,
            capsize=4,
            marker='o'
        )

    plt.title("Average Monthly Temperature by Country")
    plt.xlabel("Month")
    plt.ylabel("Average Temperature (°C)")
    plt.legend(title="Country")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("country_monthly_avg_temp.png")
    plt.close()


    # Question 5 - Fitting model for different values of `k`
    israel_df = df[df["Country"] == "Israel"]
    X = israel_df["DayOfYear"].values
    y = israel_df["Temp"].values

    # Split to train/test (75%/25%)
    rng = np.random.default_rng(seed=0)
    indices = rng.permutation(len(X))
    split_idx = int(0.75 * len(X))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Store losses for each degree
    losses = []

    print("Test errors for polynomial degrees:")
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        loss = model.loss(X_test, y_test)
        losses.append(loss)
        print(f"Degree {k}: Test Error = {loss:.2f}")

    # Bar plot of test errors
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 11), losses, color="lightcoral", edgecolor="black")
    plt.xticks(range(1, 11))
    plt.xlabel("Polynomial Degree (k)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Test Error vs. Polynomial Degree (Israel)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("polynomial_test_errors_israel.png")
    plt.close()

    # Question 6 - Evaluating fitted model on different countries

    best_k = 5
    model = PolynomialFitting(best_k)

    # Fit on ALL Israel data (not just the train split from earlier)
    israel_df = df[df["Country"] == "Israel"]
    X_israel = israel_df["DayOfYear"].values
    y_israel = israel_df["Temp"].values
    model.fit(X_israel, y_israel)

    # Evaluate on other countries
    countries = df["Country"].unique()
    losses = {}

    for country in countries:
        if country == "Israel":
            continue
        country_df = df[df["Country"] == country]
        X_country = country_df["DayOfYear"].values
        y_country = country_df["Temp"].values

        loss = model.loss(X_country, y_country)
        losses[country] = loss

    # Bar plot of test errors per country
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.bar(losses.keys(), losses.values(), color="mediumseagreen", edgecolor="black")
    plt.ylabel("MSE")
    plt.title(f"Model Error Using Israel's Degree-{best_k} Model")
    plt.tight_layout()
    plt.savefig("israel_model_generalization.png")
    plt.close()
