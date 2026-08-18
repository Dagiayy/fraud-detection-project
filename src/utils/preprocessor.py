import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class FraudPreprocessor:
    """Preprocessor for fraud detection data with Phase 1 enhancements."""

    def __init__(self, fraud_path: str, ip_path: str, credit_path: Optional[str] = None):
        self.df_fraud = pd.read_csv(fraud_path)
        self.df_ip = pd.read_csv(ip_path)
        self.df_credit = pd.read_csv(credit_path) if credit_path else None
        self.scaler = StandardScaler()

    # -------------------------
    # Basic Cleaning
    # -------------------------
    def handle_missing_values(self):
        if self.df_fraud is not None:
            self.df_fraud.fillna(self.df_fraud.median(numeric_only=True), inplace=True)
            self.df_fraud.dropna(inplace=True)
        if self.df_credit is not None:
            self.df_credit.fillna(0, inplace=True)

    def convert_datetimes(self):
        for col in ['signup_time', 'purchase_time']:
            if col in self.df_fraud.columns:
                self.df_fraud[col] = pd.to_datetime(self.df_fraud[col])
            else:
                raise KeyError(f"Column '{col}' not found in Fraud_Data.csv")

    def clean_data(self):
        if self.df_fraud is not None:
            self.df_fraud.drop_duplicates(inplace=True)
            for col in ['user_id', 'device_id']:
                if col in self.df_fraud.columns:
                    self.df_fraud.drop(columns=col, inplace=True)
        if self.df_credit is not None:
            self.df_credit.drop_duplicates(inplace=True)

    # -------------------------
    # Feature Engineering
    # -------------------------
    def add_time_features(self):
        df = self.df_fraud
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        self.df_fraud = df

    def calculate_velocity(self):
        df = self.df_fraud
        df['time_since_signup'] = df['time_since_signup'].replace(0, np.nan)
        df['spending_speed'] = df['purchase_value'] / df['time_since_signup']
        df['spending_speed'] = df['spending_speed'].fillna(0)
        self.df_fraud = df

    def add_transaction_frequency(self):
        df = self.df_fraud
        if 'user_id' in df.columns:
            df['transaction_frequency'] = df.groupby('user_id')['purchase_time'].transform('count')
        self.df_fraud = df

    # -------------------------
    # Geolocation
    # -------------------------
    def merge_country(self):
        df = self.df_fraud
        df['ip_int'] = df['ip_address'].apply(lambda x: int(x) if not pd.isna(x) else 0)
        self.df_ip['lower_bound_ip_address'] = self.df_ip['lower_bound_ip_address'].astype(int)
        self.df_ip['upper_bound_ip_address'] = self.df_ip['upper_bound_ip_address'].astype(int)

        bounds_list = list(zip(
            self.df_ip['lower_bound_ip_address'],
            self.df_ip['upper_bound_ip_address'],
            self.df_ip['country']
        ))

        def fast_country_lookup(ip):
            left, right = 0, len(bounds_list)-1
            while left <= right:
                mid = (left + right) // 2
                low, high, country = bounds_list[mid]
                if low <= ip <= high:
                    return country
                elif ip < low:
                    right = mid - 1
                else:
                    left = mid + 1
            return 'Unknown'

        df['country'] = df['ip_int'].apply(fast_country_lookup)
        df.drop(columns=['ip_int', 'ip_address'], inplace=True)
        self.df_fraud = df

    # -------------------------
    # Encoding / Scaling / Balance
    # -------------------------
    def encode_categorical(self, categorical_columns: Optional[List[str]] = None):
        df = self.df_fraud
        if categorical_columns is None:
            categorical_columns = ['source', 'browser', 'sex', 'country']
        cols_to_encode = [c for c in categorical_columns if c in df.columns]
        if cols_to_encode:
            df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
        self.df_fraud = df

    def scale_features(self, numerical_columns: List[str]):
        df = self.df_fraud
        cols = [c for c in numerical_columns if c in df.columns]
        df[cols] = self.scaler.fit_transform(df[cols])
        self.df_fraud = df

    def balance_classes(self, target_col='class', method='smote'):
        df = self.df_fraud
        X = df.drop(columns=[target_col])
        # Drop any datetime columns before SMOTE
        datetime_cols = X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        if datetime_cols:
            X = X.drop(columns=datetime_cols)
        y = df[target_col]
        sampler = SMOTE(random_state=42) if method == 'smote' else RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        self.df_fraud = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)

    # -------------------------
    # Enhanced EDA
    # -------------------------
    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12,8))
        num_cols = self.df_fraud.select_dtypes(include=np.number).columns.tolist()
        sns.heatmap(self.df_fraud[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_class_distribution(self, target_col='class'):
        sns.countplot(x=target_col, data=self.df_fraud)
        plt.title("Class Distribution")
        plt.show()

    def pca_analysis(self, n_components=10):
        if self.df_credit is not None:
            X = self.df_credit[[f"V{i}" for i in range(1,29)]]
            X_scaled = self.scaler.fit_transform(X)
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
            plt.figure(figsize=(10,6))
            plt.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), marker='o')
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title("PCA Explained Variance")
            plt.grid(True)
            plt.show()

    # -------------------------
    # Save
    # -------------------------
    def save_processed_data(self, output_path):
        self.df_fraud.to_csv(output_path, index=False)

# -------------------------
# Run full pipeline
# -------------------------
if __name__ == "__main__":
    preprocessor = FraudPreprocessor(
        fraud_path="data/raw/Fraud_Data.csv",
        ip_path="data/raw/IpAddress_to_Country.csv",
        credit_path="data/raw/creditcard.csv"
    )

    preprocessor.handle_missing_values()
    preprocessor.convert_datetimes()
    preprocessor.add_time_features()
    preprocessor.calculate_velocity()
    preprocessor.add_transaction_frequency()
    preprocessor.merge_country()
    preprocessor.clean_data()
    preprocessor.encode_categorical()
    preprocessor.scale_features(
        numerical_columns=['purchase_value','age','spending_speed','hour_of_day','day_of_week','time_since_signup']
    )
    preprocessor.balance_classes(target_col='class', method='smote')

    # Optional EDA
    preprocessor.plot_correlation_heatmap()
    preprocessor.plot_class_distribution()
    preprocessor.pca_analysis()

    preprocessor.save_processed_data("data/processed/enhanced_processed_fraud_data.csv")
