import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import time

# Rich imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich import print as rprint
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to regular print
    def rprint(*args, **kwargs):
        print(*args, **kwargs)


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

        # Rich console
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

    def load_data(self, train_path: str, test_path: str, subset_ratio: Optional[float] = None) -> 'MLPipeline':
        try:
            if RICH_AVAILABLE and self.console:
                # Display loading panel
                self.console.print(Panel.fit(
                    f"[cyan]Loading datasets...[/cyan]\n"
                    f"[dim]Train:[/dim] {train_path}\n"
                    f"[dim]Test:[/dim] {test_path}",
                    title="📊 Loading Data",
                    border_style="blue"
                ))

            # Load CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Apply subset if requested
            if subset_ratio is not None:
                if not 0 < subset_ratio <= 1:
                    raise ValueError(f"subset_ratio must be between 0 and 1, got {subset_ratio}")

                if RICH_AVAILABLE and self.console:
                    self.console.print(f"[yellow]⚠️  Applying subset ratio: {subset_ratio:.2%}[/yellow]")

                train_df = train_df.sample(frac=subset_ratio, random_state=42).reset_index(drop=True)
                test_df = test_df.sample(frac=subset_ratio, random_state=42).reset_index(drop=True)

            # Separate labels and features
            self.label_train = train_df['label'].values
            self.data_train = train_df.drop('label', axis=1).values

            self.label_test = test_df['label'].values
            self.data_test = test_df.drop('label', axis=1).values

            if RICH_AVAILABLE and self.console:
                # Create summary table
                table = Table(title="Dataset Summary", show_header=True, header_style="bold cyan")
                table.add_column("Metric", style="cyan", no_wrap=True)
                table.add_column("Value", style="magenta")

                table.add_row("Train samples", str(self.data_train.shape[0]))
                table.add_row("Test samples", str(self.data_test.shape[0]))
                table.add_row("Features per sample", str(self.data_train.shape[1]))
                if subset_ratio is not None:
                    table.add_row("Subset ratio applied", f"{subset_ratio:.2%}")

                self.console.print(table)

                # Success panel
                self.console.print(Panel(
                    "[bold green]✅ Data loaded successfully![/bold green]",
                    border_style="green"
                ))
            else:
                print(f"Data loaded successfully:")
                print(f"  Training set: {self.data_train.shape[0]} samples")
                print(f"  Test set: {self.data_test.shape[0]} samples")
                print(f"  Features per sample: {self.data_train.shape[1]}")

            return self

        except FileNotFoundError as e:
            if RICH_AVAILABLE and self.console:
                self.console.print(f"[bold red]❌ Error: CSV file not found: {e}[/bold red]")
            raise FileNotFoundError(f"CSV file not found: {e}")
        except KeyError:
            if RICH_AVAILABLE and self.console:
                self.console.print("[bold red]❌ Error: CSV file must contain a 'label' column[/bold red]")
            raise KeyError("CSV file must contain a 'label' column")

    @staticmethod
    def extract_wavelet_features(data: np.ndarray, wavelet: str = 'bior1.3', use_progress: bool = True) -> np.ndarray:
        wavelet_features = []

        if RICH_AVAILABLE and use_progress:
            iterator = track(range(data.shape[0]), description="[cyan]Extracting wavelet features...", total=data.shape[0])
        else:
            iterator = range(data.shape[0])
            if use_progress:
                print("Extracting wavelet features...")

        for i in iterator:
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

        start_time = time.time()

        if RICH_AVAILABLE and self.console:
            # Training info panel
            self.console.print(Panel.fit(
                f"[cyan]Model:[/cyan] {self.model.__class__.__name__}\n"
                f"[cyan]Use wavelets:[/cyan] {'[green]Yes[/green]' if use_wavelets else '[red]No[/red]'}\n"
                f"[cyan]Normalize:[/cyan] {'[green]Yes[/green]' if normalize else '[red]No[/red]'}",
                title="🎯 Training Model",
                border_style="blue"
            ))
            self.console.print("[cyan]Preparing features...[/cyan]")
        else:
            print(f"Preparing features (normalize={normalize}, wavelets={use_wavelets})...")

        self.features_train = self._prepare_features(
            self.data_train,
            use_wavelets=use_wavelets,
            normalize=normalize
        )

        if RICH_AVAILABLE and self.console:
            self.console.print(f"[cyan]Training model on {self.features_train.shape[0]} samples with {self.features_train.shape[1]} features...[/cyan]")
        else:
            print(f"Training model on {self.features_train.shape[0]} samples with {self.features_train.shape[1]} features...")

        self.model.fit(self.features_train, self.label_train)

        elapsed_time = time.time() - start_time

        if RICH_AVAILABLE and self.console:
            self.console.print(Panel(
                f"[bold green]✅ Model training completed in {elapsed_time:.2f}s[/bold green]",
                border_style="green"
            ))
        else:
            print(f"Model training completed in {elapsed_time:.2f}s")

        return self

    def fit_with_gridsearch(self, param_grid: Dict[str, Any], cv: int = 5,
                           scoring: str = 'accuracy', use_wavelets: bool = True,
                           normalize: bool = True, n_jobs: int = -1,
                           verbose: int = 3) -> 'MLPipeline':
        if self.data_train is None:
            raise ValueError("No training data loaded. Call load_data() first.")

        self.use_wavelets = use_wavelets
        self.normalize = normalize

        if RICH_AVAILABLE and self.console:
            # GridSearch info panel
            self.console.print(Panel.fit(
                f"[cyan]Model:[/cyan] {self.model.__class__.__name__}\n"
                f"[cyan]Cross-validation folds:[/cyan] {cv}\n"
                f"[cyan]Scoring metric:[/cyan] {scoring}\n"
                f"[cyan]Use wavelets:[/cyan] {'[green]Yes[/green]' if use_wavelets else '[red]No[/red]'}\n"
                f"[cyan]Normalize:[/cyan] {'[green]Yes[/green]' if normalize else '[red]No[/red]'}",
                title="🔍 GridSearchCV Hyperparameter Tuning",
                border_style="blue"
            ))

            # Parameter grid table
            param_table = Table(title="Parameter Grid", show_header=True, header_style="bold cyan")
            param_table.add_column("Parameter", style="cyan", no_wrap=True)
            param_table.add_column("Values", style="magenta")

            for param_name, param_values in param_grid.items():
                param_table.add_row(param_name, str(param_values))

            self.console.print(param_table)
            self.console.print("[cyan]Preparing features...[/cyan]")
        else:
            print(f"Preparing features (normalize={normalize}, wavelets={use_wavelets})...")

        self.features_train = self._prepare_features(
            self.data_train,
            use_wavelets=use_wavelets,
            normalize=normalize
        )

        if RICH_AVAILABLE and self.console:
            self.console.print(f"[cyan]Starting GridSearchCV with {cv}-fold cross-validation...[/cyan]")
        else:
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

        start_time = time.time()
        grid_search.fit(self.features_train, self.label_train)
        elapsed_time = time.time() - start_time

        # Store the best model
        self.model = grid_search.best_estimator_

        if RICH_AVAILABLE and self.console:
            # Results panel
            best_params_str = "\n".join([f"[cyan]{k}:[/cyan] [yellow]{v}[/yellow]" for k, v in grid_search.best_params_.items()])

            self.console.print(Panel.fit(
                f"[bold green]Best {scoring} score: {grid_search.best_score_:.4f}[/bold green]\n\n"
                f"[bold cyan]Best Parameters:[/bold cyan]\n{best_params_str}\n\n"
                f"[dim]Training time: {elapsed_time:.2f}s[/dim]",
                title="✅ GridSearchCV Results",
                border_style="green"
            ))

            # Detailed results table
            results_df = pd.DataFrame(grid_search.cv_results_)
            if len(results_df) <= 20:  # Only show detailed table if not too many results
                results_table = Table(title="All CV Results", show_header=True, header_style="bold cyan")
                results_table.add_column("Rank", style="cyan", no_wrap=True)
                results_table.add_column("Mean Score", style="magenta")
                results_table.add_column("Std Score", style="blue")
                results_table.add_column("Parameters", style="yellow")

                # Sort by rank
                sorted_indices = results_df['rank_test_score'].argsort()
                for idx in sorted_indices[:10]:  # Top 10 results
                    rank = results_df.iloc[idx]['rank_test_score']
                    mean_score = results_df.iloc[idx]['mean_test_score']
                    std_score = results_df.iloc[idx]['std_test_score']
                    params = results_df.iloc[idx]['params']

                    # Color based on rank
                    if rank == 1:
                        rank_str = f"[bold green]{rank}[/bold green]"
                        score_str = f"[bold green]{mean_score:.4f}[/bold green]"
                    elif rank <= 3:
                        rank_str = f"[yellow]{rank}[/yellow]"
                        score_str = f"[yellow]{mean_score:.4f}[/yellow]"
                    else:
                        rank_str = str(int(rank))
                        score_str = f"{mean_score:.4f}"

                    results_table.add_row(
                        rank_str,
                        score_str,
                        f"{std_score:.4f}",
                        str(params)
                    )

                self.console.print(results_table)
        else:
            print("\n" + "="*60)
            print("GridSearchCV Results:")
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Best {scoring} score: {grid_search.best_score_:.4f}")
            print(f"  Training time: {elapsed_time:.2f}s")
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

        if RICH_AVAILABLE and self.console:
            self.console.print(Panel.fit(
                f"[cyan]Samples:[/cyan] {self.data_test.shape[0]}\n"
                f"[cyan]Use wavelets:[/cyan] {'[green]Yes[/green]' if use_wavelets else '[red]No[/red]'}\n"
                f"[cyan]Normalize:[/cyan] {'[green]Yes[/green]' if normalize else '[red]No[/red]'}",
                title="🎲 Making Predictions",
                border_style="blue"
            ))
            self.console.print("[cyan]Preparing test features...[/cyan]")
        else:
            print(f"Preparing test features (normalize={normalize}, wavelets={use_wavelets})...")

        self.features_test = self._prepare_features(
            self.data_test,
            use_wavelets=use_wavelets,
            normalize=normalize
        )

        if RICH_AVAILABLE and self.console:
            self.console.print(f"[cyan]Making predictions on {self.features_test.shape[0]} samples...[/cyan]")
        else:
            print(f"Making predictions on {self.features_test.shape[0]} samples...")

        self.predictions = self.model.predict(self.features_test)

        if RICH_AVAILABLE and self.console:
            self.console.print(Panel(
                f"[bold green]✅ Predictions completed: {len(self.predictions)} predictions made[/bold green]",
                border_style="green"
            ))
        else:
            print("Predictions completed.")

        return self.predictions

    def evaluate(self, show_plot: bool = True, show_report: bool = True) -> np.ndarray:

        if self.predictions is None:
            raise ValueError("No predictions available. Call predict() first.")

        # Compute accuracy
        accuracy = np.mean(self.predictions == self.label_test)

        # Compute confusion matrix
        cm = confusion_matrix(self.label_test, self.predictions)

        if RICH_AVAILABLE and self.console:
            # Determine color based on accuracy
            if accuracy > 0.8:
                acc_color = "green"
                acc_emoji = "✅"
            elif accuracy > 0.6:
                acc_color = "yellow"
                acc_emoji = "⚠️"
            else:
                acc_color = "red"
                acc_emoji = "❌"

            # Evaluation header panel
            self.console.print(Panel.fit(
                f"[bold {acc_color}]{acc_emoji} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)[/bold {acc_color}]",
                title="📈 Model Evaluation",
                border_style=acc_color
            ))

        # Show classification report
        if show_report:
            if RICH_AVAILABLE and self.console:
                # Parse classification report
                from sklearn.metrics import precision_recall_fscore_support

                # Get unique classes
                classes = sorted(set(self.label_test) | set(self.predictions))

                # Calculate metrics per class
                precision, recall, f1, support = precision_recall_fscore_support(
                    self.label_test, self.predictions, labels=classes, zero_division=0
                )

                # Create classification report table
                report_table = Table(title="Classification Report", show_header=True, header_style="bold cyan")
                report_table.add_column("Class", style="cyan", no_wrap=True, justify="center")
                report_table.add_column("Precision", justify="center")
                report_table.add_column("Recall", justify="center")
                report_table.add_column("F1-Score", justify="center")
                report_table.add_column("Support", style="blue", justify="center")

                def get_metric_color(value):
                    """Return color based on metric value"""
                    if value > 0.8:
                        return "green"
                    elif value > 0.6:
                        return "yellow"
                    else:
                        return "red"

                # Add rows for each class
                for i, cls in enumerate(classes):
                    prec_color = get_metric_color(precision[i])
                    rec_color = get_metric_color(recall[i])
                    f1_color = get_metric_color(f1[i])

                    report_table.add_row(
                        str(cls),
                        f"[{prec_color}]{precision[i]:.4f}[/{prec_color}]",
                        f"[{rec_color}]{recall[i]:.4f}[/{rec_color}]",
                        f"[{f1_color}]{f1[i]:.4f}[/{f1_color}]",
                        str(support[i])
                    )

                # Add average rows
                macro_precision = np.mean(precision)
                macro_recall = np.mean(recall)
                macro_f1 = np.mean(f1)
                weighted_precision = np.average(precision, weights=support)
                weighted_recall = np.average(recall, weights=support)
                weighted_f1 = np.average(f1, weights=support)

                report_table.add_section()
                report_table.add_row(
                    "[bold]Macro Avg[/bold]",
                    f"[bold]{macro_precision:.4f}[/bold]",
                    f"[bold]{macro_recall:.4f}[/bold]",
                    f"[bold]{macro_f1:.4f}[/bold]",
                    str(np.sum(support))
                )
                report_table.add_row(
                    "[bold]Weighted Avg[/bold]",
                    f"[bold]{weighted_precision:.4f}[/bold]",
                    f"[bold]{weighted_recall:.4f}[/bold]",
                    f"[bold]{weighted_f1:.4f}[/bold]",
                    str(np.sum(support))
                )

                self.console.print(report_table)
            else:
                print("\n" + "="*60)
                print("Classification Report:")
                print("="*60)
                print(classification_report(self.label_test, self.predictions))
                print("="*60 + "\n")

        # Show confusion matrix plot
        if show_plot:
            if RICH_AVAILABLE and self.console:
                self.console.print("\n[bold cyan]Confusion Matrix:[/bold cyan]")

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

            if RICH_AVAILABLE:
                plt.title(f'Matrice de confusion - Accuracy: {accuracy:.4f}', fontsize=14, fontweight='bold')
            else:
                plt.title('Matrice de confusion', fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.show()

        return cm

    def get_accuracy(self, verbose: bool = False) -> float:

        if self.predictions is None:
            raise ValueError("No predictions available. Call predict() first.")

        accuracy = np.mean(self.predictions == self.label_test)

        if verbose:
            if RICH_AVAILABLE and self.console:
                # Determine color based on accuracy
                if accuracy > 0.8:
                    acc_color = "green"
                    acc_emoji = "✅"
                elif accuracy > 0.6:
                    acc_color = "yellow"
                    acc_emoji = "⚠️"
                else:
                    acc_color = "red"
                    acc_emoji = "❌"

                self.console.print(Panel(
                    f"[bold {acc_color}]{acc_emoji} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)[/bold {acc_color}]",
                    border_style=acc_color
                ))
            else:
                print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        return accuracy

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return (f"MLPipeline(model={self.model.__class__.__name__}, "
                f"wavelet='{self.wavelet}', "
                f"trained={self.features_train is not None})")
