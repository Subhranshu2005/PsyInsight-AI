"""
PsyInsight AI
Machine Learning Classification Module

Author: Subhranshu Ranjan Sahoo

This module provides a unified interface for
classification algorithms.
"""

from typing import Optional

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class PsyClassifier:
    """
    Unified Classification Engine.

    Supports:

    • Logistic Regression
    • Decision Tree
    • Random Forest
    • Support Vector Machine
    • K-Nearest Neighbors
    • Gaussian Naive Bayes
    """

    def __init__(self):
        """
        Initialize the classifier.
        """

        self.model = None
        self.model_name = None

    ##################################################
    # Logistic Regression
    ##################################################

    def logistic_regression(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
    ):
        """
        Initialize Logistic Regression.
        """

        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
        )

        self.model_name = "Logistic Regression"

        return self

    ##################################################
    # Decision Tree
    ##################################################

    def decision_tree(
        self,
        random_state: int = 42,
    ):
        """
        Initialize Decision Tree.
        """

        self.model = DecisionTreeClassifier(
            random_state=random_state,
        )

        self.model_name = "Decision Tree"

        return self

    ##################################################
    # Random Forest
    ##################################################

    def random_forest(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize Random Forest.
        """

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )

        self.model_name = "Random Forest"

        return self

    ##################################################
    # Support Vector Machine
    ##################################################

    def support_vector_machine(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
    ):
        """
        Initialize Support Vector Machine.
        """

        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
        )

        self.model_name = "Support Vector Machine"

        return self

    ##################################################
    # K Nearest Neighbors
    ##################################################

    def knn(
        self,
        neighbors: int = 5,
    ):
        """
        Initialize KNN.
        """

        self.model = KNeighborsClassifier(
            n_neighbors=neighbors,
        )

        self.model_name = "K Nearest Neighbors"

        return self

    ##################################################
    # Gaussian Naive Bayes
    ##################################################

    def naive_bayes(self):
        """
        Initialize Gaussian Naive Bayes.
        """

        self.model = GaussianNB()

        self.model_name = "Gaussian Naive Bayes"

        return self

    ##################################################
    # Information
    ##################################################

    def current_model(self) -> Optional[str]:
        """
        Return the currently selected model.
        """

        return self.model_name

    ##################################################
    # Save Model
    ##################################################

    def save_model(
        self,
        filename: str,
    ):
        """
        Save trained model.
        """

        if self.model is None:
            raise ValueError(
                "No model has been initialized."
            )

        joblib.dump(
            self.model,
            filename,
        )

    ##################################################
    # Load Model
    ##################################################

    def load_model(
        self,
        filename: str,
    ):
        """
        Load a trained model.
        """

        self.model = joblib.load(
            filename
        )

        return self
