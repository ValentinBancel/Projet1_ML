import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


class MLPipeline:
    def __init__(self, model: Any, wavelet: str = 'bior1.3'):
        self.model = model
        self.wavelet = wavelet
        self.use_wavelets = False
        self.normalize = False

        # Data containers
        self.data_train = None
        self.label_train = None
        self.data_test = None
        self.label_test = None

        # Transformed features
        self.features_train = None
        self.features_test = None

        # Predictions
        self.predictions = None

    def load_data(self, train_path: str, test_path: str, subset_ratio: Optional[float] = None) -> 'MLPipeline':
        try:
            # Load CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Apply subset if requested
            if subset_ratio is not None:
                if not 0 < subset_ratio <= 1:
                    raise ValueError(f"subset_ratio must be between 0 and 1, got {subset_ratio}")
                train_df = train_df.sample(frac=subset_ratio, random_state=42).reset_index(drop=True)
                test_df = test_df.sample(frac=subset_ratio, random_state=42).reset_index(drop=True)

            # Separate labels and features
            self.label_train = train_df['label'].values
            self.data_train = train_df.drop('label', axis=1).values

            self.label_test = test_df['label'].values
            self.data_test = test_df.drop('label', axis=1).values

            print(f"Data loaded successfully:")
            print(f"  Training set: {self.data_train.shape[0]} samples")
            print(f"  Test set: {self.data_test.shape[0]} samples")
            print(f"  Features per sample: {self.data_train.shape[1]}")

            return self

        except FileNotFoundError as e:
            raise FileNotFoundError(f"CSV file not found: {e}")
        except KeyError:
            raise KeyError("CSV file must contain a 'label' column")

    @staticmethod
    def extract_wavelet_features(data: np.ndarray, wavelet: str = 'bior1.3') -> np.ndarray:
        wavelet_features = []

        for i in range(data.shape[0]):
            # Reshape flat vector to 28x28 image
            image = data[i].reshape(28, 28)

            # Apply 2D Discrete Wavelet Transform
            coeffs = pywt.dwt2(image, wavelet)
            cA, (cH, cV, cD) = coeffs  # LL, (LH, HL, HH)

            # Flatten and concatenate all subbands
            features = np.concatenate([
                cA.flatten(),  # LL (approximation)
                cH.flatten(),  # LH (horizontal details)
                cV.flatten(),  # HL (vertical details)
                cD.flatten()   # HH (diagonal details)
            ])

            wavelet_features.append(features)

        return np.array(wavelet_features)

    def _prepare_features(self, data: np.ndarray, use_wavelets: bool = True,
                         normalize: bool = True) -> np.ndarray:
        features = data.copy()

        # Normalize if requested
        if normalize:
            features = features / 255.0

        # Apply wavelet transform if requested
        if use_wavelets:
            features = self.extract_wavelet_features(features, self.wavelet)

        return features

    def fit(self, use_wavelets: bool = True, normalize: bool = True) -> 'MLPipeline':
        if self.data_train is None:
            raise ValueError("No training data loaded. Call load_data() first.")

        self.use_wavelets = use_wavelets
        self.normalize = normalize

        print(f"Preparing features (normalize={normalize}, wavelets={use_wavelets})...")
        self.features_train = self._prepare_features(
            self.data_train,
            use_wavelets=use_wavelets,
            normalize=normalize
        )

        print(f"Training model on {self.features_train.shape[0]} samples with {self.features_train.shape[1]} features...")
        self.model.fit(self.features_train, self.label_train)
        print("Model training completed.")

        return self

    def fit_with_gridsearch(self, param_grid: Dict[str, Any], cv: int = 5,
                           scoring: str = 'accuracy', use_wavelets: bool = True,
                           normalize: bool = True, n_jobs: int = -1,
                           verbose: int = 2) -> 'MLPipeline':
        if self.data_train is None:
            raise ValueError("No training data loaded. Call load_data() first.")

        self.use_wavelets = use_wavelets
        self.normalize = normalize

        print(f"Preparing features (normalize={normalize}, wavelets={use_wavelets})...")
        self.features_train = self._prepare_features(
            self.data_train,
            use_wavelets=use_wavelets,
            normalize=normalize
        )

        print(f"Starting GridSearchCV with {cv}-fold cross-validation...")
        print(f"Parameter grid: {param_grid}")

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )

        grid_search.fit(self.features_train, self.label_train)

        # Store the best model
        self.model = grid_search.best_estimator_

        print("\n" + "="*60)
        print("GridSearchCV Results:")
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best {scoring} score: {grid_search.best_score_:.4f}")
        print("="*60 + "\n")

        return self

    def predict(self, use_wavelets: Optional[bool] = None,
                normalize: Optional[bool] = None) -> np.ndarray:

        if self.data_test is None:
            raise ValueError("No test data loaded. Call load_data() first.")

        # Use same settings as training if not specified
        if use_wavelets is None:
            use_wavelets = self.use_wavelets
        if normalize is None:
            normalize = self.normalize

        print(f"Preparing test features (normalize={normalize}, wavelets={use_wavelets})...")
        self.features_test = self._prepare_features(
            self.data_test,
            use_wavelets=use_wavelets,
            normalize=normalize
        )

        print(f"Making predictions on {self.features_test.shape[0]} samples...")
        self.predictions = self.model.predict(self.features_test)
        print("Predictions completed.")

        return self.predictions

    def evaluate(self, show_plot: bool = True, show_report: bool = True) -> np.ndarray:

        if self.predictions is None:
            raise ValueError("No predictions available. Call predict() first.")

        # Compute confusion matrix
        cm = confusion_matrix(self.label_test, self.predictions)

        # Show classification report
        if show_report:
            print("\n" + "="*60)
            print("Classification Report:")
            print("="*60)
            print(classification_report(self.label_test, self.predictions))
            print("="*60 + "\n")

        # Show confusion matrix plot
        if show_plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='plasma',
                xticklabels=range(10),
                yticklabels=range(10)
            )
            plt.xlabel('Prédiction', fontsize=12)
            plt.ylabel('Réalité', fontsize=12)
            plt.title('Matrice de confusion', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        return cm

    def get_accuracy(self) -> float:

        if self.predictions is None:
            raise ValueError("No predictions available. Call predict() first.")

        accuracy = np.mean(self.predictions == self.label_test)
        return accuracy

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return (f"MLPipeline(model={self.model.__class__.__name__}, "
                f"wavelet='{self.wavelet}', "
                f"trained={self.features_train is not None})")
