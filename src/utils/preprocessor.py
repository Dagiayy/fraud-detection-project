# src/utils/preprocessor.py

from imblearn.over_sampling import SMOTE  # type: ignore
from imblearn.under_sampling import RandomUnderSampler  # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path


def balance_classes(X, y, method='smote'):
    """
    Handles class imbalance using the specified method.

    Args:
        X (DataFrame or ndarray): Features.
        y (Series or ndarray): Target labels.
        method (str): 'smote' or 'undersample'

    Returns:
        X_res, y_res: Resampled feature set and labels.
    """
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    elif method == 'undersample':
        undersample = RandomUnderSampler(random_state=42)
        X_res, y_res = undersample.fit_resample(X, y)
    else:
        raise ValueError("Method must be either 'smote' or 'undersample'.")

    return X_res, y_res


def scale_features(X, method='standard'):
    """
    Scales numerical features using StandardScaler or MinMaxScaler.

    Args:
        X (DataFrame or ndarray): Data to scale.
        method (str): 'standard' or 'minmax'

    Returns:
        ndarray: Scaled data
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def encode_categorical(X, categorical_columns=None, drop_first=True):
    """
    Applies One-Hot Encoding to specified categorical columns only if they exist.

    Args:
        X (DataFrame): Input features.
        categorical_columns (list or None): List of categorical columns to encode. If None, uses default low-cardinality columns.
        drop_first (bool): Whether to drop first dummy column.

    Returns:
        DataFrame: Encoded data.
    """
    if categorical_columns is None:
        # Default low-cardinality columns to encode
        categorical_columns = ['source', 'browser', 'sex']

    # Keep only columns present in X
    cols_to_encode = [col for col in categorical_columns if col in X.columns]
    if not cols_to_encode:
        print("⚠️ No categorical columns to encode.")
        return X

    X_encoded = pd.get_dummies(X, columns=cols_to_encode, drop_first=drop_first)
    return X_encoded



if __name__ == "__main__":
    raw_data_path = Path("data/processed/processed_fraud_data.csv")
    df = pd.read_csv(raw_data_path)
    print("✅ Raw data loaded. Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    high_cardinality_cols = ['user_id', 'device_id', 'ip_address']

    # Drop columns only if they exist
    cols_to_drop = ["class"] + [col for col in high_cardinality_cols if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df["class"]

    print("Target distribution before balancing:\n", y.value_counts())

    # Convert datetime columns to numeric (unix timestamp) if they exist
    for dt_col in ['signup_time', 'purchase_time']:
        if dt_col in X.columns:
            X[dt_col] = pd.to_datetime(X[dt_col]).astype(int) / 10**9

    # Encode categorical features
    X_encoded = encode_categorical(X)
    print("✅ Encoding done. Encoded shape:", X_encoded.shape)

    # Scale features
    X_scaled = scale_features(X_encoded, method="standard")
    print("✅ Scaling done. Example row:\n", X_scaled[0])

    # Balance classes using SMOTE
    X_balanced, y_balanced = balance_classes(X_scaled, y, method="smote")
    print("✅ Class balancing done. New target distribution:\n", pd.Series(y_balanced).value_counts())

    # Save processed data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_df = pd.DataFrame(X_balanced, columns=X_encoded.columns)
    processed_df["class"] = y_balanced

    save_path = processed_dir / "processed_fraud_data.csv"
    processed_df.to_csv(save_path, index=False)
    print(f"✅ Processed data saved to {save_path}")

    print("\n🎉 Preprocessing pipeline test completed successfully.")
