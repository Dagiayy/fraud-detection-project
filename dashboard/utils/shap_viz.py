# dashboard/utils/shap_viz.py
from __future__ import annotations

import logging
from typing import Optional, List
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ShapVisualizer:
    """
    SHAP visualizations for fraud model explanations.

    Features:
        - Summary plot (global feature importance)
        - Dependence plot (feature interaction)
        - Force plot (local explanation per transaction)
        - Optional interactive mode for stakeholder-friendly UI
    """

    def __init__(self, model, X: pd.DataFrame):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained tree-based model (LightGBM/XGBoost/RandomForest)
            X: Training or reference dataset for SHAP baseline
        """
        self.model = model
        self.X = X.copy()
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(self.X)
        logger.info("SHAP explainer initialized successfully.")

    def summary_plot(self, interactive: bool = False) -> None:
        """Generate SHAP summary plot for global feature importance."""
        st.subheader("SHAP Summary Plot")
        if interactive:
            shap_values_mean = np.abs(self.shap_values[1]).mean(axis=0)
            feature_importance = pd.DataFrame({
                "Feature": self.X.columns,
                "Importance": shap_values_mean
            }).sort_values(by="Importance", ascending=True)
            fig = px.bar(feature_importance, x="Importance", y="Feature", orientation="h",
                         title="SHAP Feature Importance (Interactive)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(self.shap_values[1], self.X, show=False)
            st.pyplot(fig)

    def dependence_plot(self, feature: str, interactive: bool = False) -> None:
        """
        Generate SHAP dependence plot for a single feature.

        Args:
            feature: Feature name to show interaction effect
            interactive: If True, use Plotly for interactive plotting
        """
        if feature not in self.X.columns:
            st.warning(f"Feature '{feature}' not found in dataset.")
            return

        st.subheader(f"SHAP Dependence Plot: {feature}")
        if interactive:
            shap_df = pd.DataFrame(self.shap_values[1], columns=self.X.columns)
            fig = px.scatter(
                self.X,
                x=feature,
                y=shap_df[feature],
                color=shap_df[feature],
                labels={"x": feature, "y": "SHAP value"},
                title=f"Interactive SHAP Dependence: {feature}",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(feature, self.shap_values[1], self.X, show=False)
            st.pyplot(fig)

    def force_plot(self, row_idx: int = 0, interactive: bool = False) -> None:
        """
        Generate SHAP force plot for a specific transaction.

        Args:
            row_idx: Index of the transaction in X
            interactive: If True, render force plot in browser
        """
        if row_idx >= len(self.X):
            st.warning(f"Row index {row_idx} out of bounds. Max index: {len(self.X)-1}")
            return

        st.subheader(f"SHAP Force Plot for Transaction {row_idx}")
        if interactive:
            shap.initjs()
            st_shap_html = shap.force_plot(
                self.explainer.expected_value[1],
                self.shap_values[1][row_idx, :],
                self.X.iloc[[row_idx]]
            )._repr_html_()
            st.components.v1.html(st_shap_html, height=400)
        else:
            fig = shap.force_plot(
                self.explainer.expected_value[1],
                self.shap_values[1][row_idx, :],
                self.X.iloc[[row_idx]],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig)

    def batch_force_plots(self, indices: Optional[List[int]] = None, interactive: bool = False) -> None:
        """
        Generate multiple force plots for a batch of transactions.

        Args:
            indices: list of row indices to plot; defaults to first 5 rows
            interactive: If True, render force plots interactively
        """
        if indices is None:
            indices = list(range(min(5, len(self.X))))
        for idx in indices:
            self.force_plot(row_idx=idx, interactive=interactive)
