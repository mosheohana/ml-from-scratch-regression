
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
CUR_YEAR = 2025


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    Preprocess training data:
    - Drop irrelevant features
    - Remove missing or invalid values
    - Add engineered features
    """
    # Drop irrelevant columns
    X = X.drop(columns=["id", "date", "lat", "long"])

    # Drop rows with missing values
    missing_mask = X.isnull().any(axis=1) | y.isnull()
    X = X[~missing_mask]
    y = y[~missing_mask]

    # Remove rows with clearly invalid values
    valid_mask = (
        (X["bedrooms"] > 0) &
        (X["bathrooms"] > 0) &
        (X["sqft_living"] > 0) &
        (X["sqft_lot"] > 0) &
        (X["floors"] > 0)
    )
    X = X[valid_mask]
    y = y[valid_mask]

    # Remove rows with any negative numeric value
    numeric_only = X.select_dtypes(include=[np.number])
    non_negative_mask = (numeric_only >= 0).all(axis=1)
    X = X[non_negative_mask]
    y = y[non_negative_mask]

    # Add engineered features
    X["house_age"] = CUR_YEAR - X["yr_built"]
    X["was_renovated"] = (X["yr_renovated"] > 0).astype(int)
    X["years_since_renovation"] = np.where(
        X["yr_renovated"] > 0,
        CUR_YEAR - X["yr_renovated"],
        CUR_YEAR - X["yr_built"]
    )
    #Ratio of living space to lot size
    X["living_to_lot_ratio"] = X["sqft_living"] / (X["sqft_lot"] + 1)

    #Ratio of above ground area to total living area
    X["above_ground_ratio"] = X["sqft_above"] / (X["sqft_living"] + 1)

    #Ratio of basement space
    X["basement_ratio"] = X["sqft_basement"] / (X["sqft_living"] +1)

    #Interaction between bathrooms and bedrooms
    X["bath_bed_ratio"] = X["bathrooms"] / (X["bedrooms"] +1)



    return X, y



def preprocess_test(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess test data:
    - Must not remove rows
    - Must match training feature structure using global TRAIN_COLUMNS
    """
    global TRAIN_COLUMNS

    # Drop irrelevant features
    X = X.drop(columns=["id", "date", "lat", "long"])

    # Add engineered features
    X["house_age"] = CUR_YEAR - X["yr_built"]
    X["was_renovated"] = (X["yr_renovated"] > 0).astype(int)
    X["years_since_renovation"] = np.where(
        X["yr_renovated"] > 0,
        CUR_YEAR - X["yr_renovated"],
        CUR_YEAR - X["yr_built"]
    )
    # Ratio of living space to lot size
    X["living_to_lot_ratio"] = X["sqft_living"] / (X["sqft_lot"] + 1)

    # Ratio of above ground area to total living area
    X["above_ground_ratio"] = X["sqft_above"] / (X["sqft_living"] + 1)

    # Ratio of basement space
    X["basement_ratio"] = X["sqft_basement"] / (X["sqft_living"] + 1)

    # Interaction between bathrooms and bedrooms
    X["bath_bed_ratio"] = X["bathrooms"] / (X["bedrooms"] + 1)


    # Add missing columns (from training)
    missing_cols = set(TRAIN_COLUMNS) - set(X.columns)
    for col in missing_cols:
        X[col] = 0

    # Drop extra columns (not in training)
    extra_cols = set(X.columns) - set(TRAIN_COLUMNS)
    if extra_cols:
        X = X.drop(columns=extra_cols)

    # Reorder columns to match training
    X = X[TRAIN_COLUMNS]

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "."):
    """
    Evaluate and plot Pearson correlation between each feature and the response.

    Parameters
    ----------
    X : pd.DataFrame
        Preprocessed training features
    y : pd.Series
        Target variable (price)
    output_path : str
        Folder where plots will be saved (assumed to exist)
    """
    for feature in X.columns:
        x_vals = X[feature].values
        y_vals = y.values

        # Safely calculate std before computing covariance
        std_x = np.std(x_vals, ddof=1)
        std_y = np.std(y_vals, ddof=1)

        if std_x == 0 or std_y == 0:
            pearson_corr = float("nan")  # Avoid division by zero

        else:
            covariance = np.cov(x_vals, y_vals)[0, 1]
            pearson_corr = covariance / (std_x * std_y)


        # Create and save the plot
        plt.figure()
        plt.scatter(x_vals, y_vals, alpha=0.5)
        plt.title(f"{feature} vs. Price\nPearson Correlation = {pearson_corr:.3f}")
        plt.xlabel(feature)
        plt.ylabel("Price")
        plt.tight_layout()

        plt.savefig(f"{output_path}/{feature}.png")
        plt.close()


def split_data(X, y):
    """
    Split X and y into training (75%) and testing (25%) sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """
    # Combine X and y into one DataFrame for synchronized shuffling
    data = X.copy()
    data["__target__"] = y

    # Shuffle the data
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)

    # Split index
    split_idx = int(0.75 * len(data))
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    # Separate back to X and y
    X_train = train.drop(columns="__target__")
    y_train = train["__target__"]
    X_test = test.drop(columns="__target__")
    y_test = test["__target__"]


    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = split_data(X, y)


    # Question 3 - preprocessing of housing prices train dataset
    X_train_clean, y_train_clean = preprocess_train(X_train, y_train)

    TRAIN_COLUMNS = X_train_clean.columns


    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train_clean, y_train_clean, output_path="feature_plots")

    # Question 5 - preprocess the test data
    X_test_clean = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages = list(range(10, 101))
    mean_losses = []
    std_losses = []

    for p in percentages:
        losses = []
        for _ in range(10):
            sample_X = X_train_clean.sample(frac=p / 100, random_state=None)
            sample_y = y_train_clean.loc[sample_X.index]

            model = LinearRegression(include_intercept=True)
            model.fit(sample_X.to_numpy(), sample_y.to_numpy())

            loss = model.loss(X_test_clean.to_numpy(), y_test.to_numpy())
            losses.append(loss)

        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)

    # Plot learning curve
    plt.figure()
    plt.plot(percentages, mean_losses, label="Mean Test Loss")
    plt.fill_between(
        percentages,
        mean_losses - 2 * std_losses,
        mean_losses + 2 * std_losses,
        alpha=0.2,
        label="±2 Std Dev"
    )
    plt.xlabel("Percentage of Training Data Used")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.close()
