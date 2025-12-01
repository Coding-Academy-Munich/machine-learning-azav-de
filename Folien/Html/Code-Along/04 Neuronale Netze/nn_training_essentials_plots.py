"""
Plotting functions for Neural Network Training Essentials slides.

All functions encapsulate data generation and plotting logic to minimize
code shown in slides. Most functions can be called without parameters.
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings
from numpy.random import f
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


# =============================================================================
# Data Generation (internal use)
# =============================================================================

def _generate_grade_data(n_samples=50, random_state=42):
    """Generate student grade prediction data."""
    np.random.seed(random_state)
    hours = np.linspace(0, 10, n_samples)
    grades = 1.0 + 0.5 * hours + 0.05 * hours**2 + np.random.normal(0, 0.5, n_samples)
    grades = np.clip(grades, 1, 6)
    X = hours.reshape(-1, 1)
    return X, grades


def _generate_loss_landscape_with_local_minimum():
    """Generate a 1D loss landscape with local and global minima."""
    w = np.linspace(-5, 5, 200)
    # Local valley (Funtensee) at w=-2, global valley at w=3
    valley_local = -2.0 * np.exp(-((w + 2) ** 2) / 1.0)
    valley_global = -3.0 * np.exp(-((w - 3) ** 2) / 1.5)
    loss = 5.5 + valley_local + valley_global
    return w, loss


# =============================================================================
# Loss and Learning Visualizations
# =============================================================================

def plot_error_during_training():
    """
    Show how error decreases during training.
    Demonstrates that more epochs lead to lower error.
    """
    X, y = _generate_grade_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    epoch_counts = [1, 5, 10, 50, 100, 500, 1000]
    train_errors = []
    test_errors = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for max_iter in epoch_counts:
            mlp = MLPRegressor(
                hidden_layer_sizes=(10,),
                activation='relu',
                max_iter=max_iter,
                random_state=42,
                solver='sgd'
            )
            mlp.fit(X_train, y_train)

            train_pred = mlp.predict(X_train)
            test_pred = mlp.predict(X_test)

            train_mse = np.mean((y_train - train_pred) ** 2)
            test_mse = np.mean((y_test - test_pred) ** 2)

            train_errors.append(train_mse)
            test_errors.append(test_mse)

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_counts, train_errors, 'o-', label='Training Error', linewidth=2)
    plt.plot(epoch_counts, test_errors, 's-', label='Test Error', linewidth=2)
    plt.xlabel('Number of Epochs (max_iter)')
    plt.ylabel('Mean Squared Error')
    plt.title('Error Decreases with Training')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions_improving():
    """
    Show predictions at different training stages (1, 10, 100, 1000 epochs).
    Visualizes how the model learns over time.
    """
    X, y = _generate_grade_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    x_plot = np.linspace(0, 10, 100).reshape(-1, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mlp_1 = MLPRegressor(
            hidden_layer_sizes=(500, 500), max_iter=1, random_state=42, solver='adam'
        )
        mlp_1.fit(X_train, y_train)
        y_pred_1 = mlp_1.predict(x_plot)

        mlp_10 = MLPRegressor(
            hidden_layer_sizes=(500, 500), max_iter=10, random_state=42, solver='adam'
        )
        mlp_10.fit(X_train, y_train)
        y_pred_10 = mlp_10.predict(x_plot)

        mlp_100 = MLPRegressor(
            hidden_layer_sizes=(500, 500), max_iter=100, random_state=42, solver='adam'
        )
        mlp_100.fit(X_train, y_train)
        y_pred_100 = mlp_100.predict(x_plot)

        mlp_1000 = MLPRegressor(
            hidden_layer_sizes=(500, 500), max_iter=1000, random_state=42, solver='adam'
        )
        mlp_1000.fit(X_train, y_train)
        y_pred_1000 = mlp_1000.predict(x_plot)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    titles = ['After 1 Epoch', 'After 10 Epochs', 'After 100 Epochs', 'After 1000 Epochs']
    predictions = [y_pred_1, y_pred_10, y_pred_100, y_pred_1000]

    for ax, title, y_pred in zip(axes, titles, predictions):
        ax.scatter(X_train, y_train, alpha=0.6, label='Training Data')
        ax.plot(x_plot, y_pred, 'r-', linewidth=2, label='Prediction')
        ax.set_title(title)
        ax.set_xlabel('Study Hours')
        ax.set_ylabel('Grade')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Gradient Descent Visualizations
# =============================================================================

def plot_simple_loss_landscape():
    """
    Plot a simple 1D parabolic loss function.
    Demonstrates the concept of a loss landscape with a minimum.
    """
    w = np.linspace(-5, 5, 100)
    loss = (w - 2) ** 2 + 1

    plt.figure(figsize=(10, 6))
    plt.plot(w, loss, 'b-', linewidth=2)
    plt.xlabel('Parameter w')
    plt.ylabel('Loss')
    plt.title('Simple Loss Landscape')
    plt.grid(True, alpha=0.3)
    plt.plot(2, 1, 'r*', markersize=20, label='Minimum (w=2)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_gradient_descent_steps():
    """
    Visualize gradient descent step-by-step on a simple parabola.
    Shows the path from start to minimum.
    """
    w = np.linspace(-5, 5, 100)
    loss = (w - 2) ** 2 + 1

    # Gradient descent simulation
    w_start = -4
    learning_rate = 0.3
    steps = 15

    w_history = [w_start]
    loss_history = [(w_start - 2) ** 2 + 1]

    w_current = w_start
    for _ in range(steps):
        grad = 2 * (w_current - 2)  # Derivative of (w-2)^2 + 1
        w_current = w_current - learning_rate * grad
        w_history.append(w_current)
        loss_history.append((w_current - 2) ** 2 + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(w, loss, 'b-', linewidth=2, label='Loss Function')
    plt.plot(w_history, loss_history, 'ro-', markersize=8, label='Gradient Descent Path')
    plt.plot(w_history[0], loss_history[0], 'g*', markersize=20, label='Start')
    plt.plot(w_history[-1], loss_history[-1], 'r*', markersize=20, label='End')

    plt.xlabel('Parameter w')
    plt.ylabel('Loss')
    plt.title('Gradient Descent: Step by Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_2d_loss_landscape():
    """
    Plot a 2D loss landscape as a contour plot.
    Shows loss as a function of two parameters.
    """
    w1_range = np.linspace(-3, 3, 100)
    w2_range = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Loss_2d = (W1 - 1) ** 2 + (W2 + 0.5) ** 2 + 0.5

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(W1, W2, Loss_2d, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Loss')
    plt.plot(1, -0.5, 'r*', markersize=20, label='Minimum')

    plt.xlabel('Parameter $w_1$')
    plt.ylabel('Parameter $w_2$')
    plt.title('2D Loss Landscape (Contour Plot)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_2d_loss_landscape_3d():
    """
    Plot a 2D loss landscape as a 3D surface.
    Provides intuitive visualization of the "mountain" to descend.
    """
    from mpl_toolkits.mplot3d import Axes3D

    w1_range = np.linspace(-3, 3, 100)
    w2_range = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Loss_2d = (W1 - 1) ** 2 + (W2 + 0.5) ** 2 + 0.5

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W1, W2, Loss_2d, cmap='viridis', alpha=0.8)
    ax.scatter([1], [-0.5], [0.5], color='red', s=200, marker='*', label='Minimum')

    ax.set_xlabel('Parameter w₁')
    ax.set_ylabel('Parameter w₂')
    ax.set_zlabel('Loss')
    ax.set_title('2D Loss Landscape (3D View)')
    plt.colorbar(surf, label='Loss', shrink=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_funtensee_local_minimum():
    """
    Plot loss landscape with local minimum (Funtensee analogy).
    Demonstrates the difference between local and global minima.
    """
    w, loss = _generate_loss_landscape_with_local_minimum()

    plt.figure(figsize=(12, 7))
    plt.plot(w, loss, 'b-', linewidth=2)

    # Mark global minimum
    global_min_idx = np.argmin(loss)
    plt.plot(w[global_min_idx], loss[global_min_idx],
             'g*', markersize=20, label='Global Minimum (Deepest Valley)')

    # Find local minimum in left region
    left_mask = w < 0.5
    left_w = w[left_mask]
    left_loss = loss[left_mask]
    local_min_idx = np.argmin(left_loss)
    local_min_w = left_w[local_min_idx]
    local_min_loss = left_loss[local_min_idx]

    plt.plot(local_min_w, local_min_loss,
             'r*', markersize=20, label='Local Minimum (Funtensee)')

    plt.xlabel('Parameter w')
    plt.ylabel('Loss')
    plt.title('Loss Landscape with Local Minimum')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.annotate('Funtensee\n(lokales Minimum / local minimum)',
                 xy=(local_min_w, local_min_loss),
                 xytext=(-2, 5),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))

    plt.annotate('Globales Minimum\nGlobal Minimum',
                 xy=(w[global_min_idx], loss[global_min_idx]),
                 xytext=(3.5, 3),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))

    plt.tight_layout()
    plt.show()


def plot_starting_point_matters():
    """
    Show how different starting points lead to different minima.
    Demonstrates why initialization matters.
    """
    w, loss = _generate_loss_landscape_with_local_minimum()

    def gradient_landscape(w_val):
        h = 0.01
        w_idx = np.argmin(np.abs(w - w_val))
        if w_idx == 0 or w_idx == len(w) - 1:
            return 0
        return (loss[w_idx + 1] - loss[w_idx - 1]) / (2 * h)

    def run_gd(w_start, lr=0.05, steps=100):
        path = [w_start]
        w_current = w_start
        for _ in range(steps):
            grad = gradient_landscape(w_current)
            w_current = w_current - lr * grad
            path.append(w_current)
        return path

    path_left = run_gd(-4.0)
    path_right = run_gd(0.5)

    plt.figure(figsize=(12, 7))
    plt.plot(w, loss, 'b-', linewidth=2, label='Loss Landscape')

    loss_left = [loss[np.argmin(np.abs(w - p))] for p in path_left]
    loss_right = [loss[np.argmin(np.abs(w - p))] for p in path_right]

    plt.plot(path_left, loss_left, 'ro-', markersize=4, alpha=0.6,
             label='Start Left → Funtensee')
    plt.plot(path_right, loss_right, 'go-', markersize=4, alpha=0.6,
             label='Start Right → Global Min')

    plt.plot(path_left[0], loss_left[0], 'r^', markersize=15, label='Start Left')
    plt.plot(path_right[0], loss_right[0], 'g^', markersize=15, label='Start Right')

    plt.xlabel('Parameter w')
    plt.ylabel('Loss')
    plt.title('Different Starting Points Lead to Different Minima')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_learning_rate_effects():
    """
    Compare different learning rates on gradient descent.
    Shows too small, good, and too large learning rates.
    """
    w, loss = _generate_loss_landscape_with_local_minimum()

    def gradient_landscape(w_val):
        h = 0.01
        w_idx = np.argmin(np.abs(w - w_val))
        if w_idx == 0 or w_idx == len(w) - 1:
            return 0
        return (loss[w_idx + 1] - loss[w_idx - 1]) / (2 * h)

    def run_gd(w_start, lr, steps=100):
        path = [w_start]
        w_current = w_start
        for _ in range(steps):
            grad = gradient_landscape(w_current)
            w_current = w_current - lr * grad
            path.append(w_current)
        return path

    path_small = run_gd(-4.0, lr=0.01, steps=100)
    path_good = run_gd(-4.0, lr=0.05, steps=100)
    path_large = run_gd(-4.0, lr=0.2, steps=100)

    plt.figure(figsize=(12, 7))
    plt.plot(w, loss, 'b-', linewidth=2, alpha=0.3)

    loss_small = [loss[np.argmin(np.abs(w - p))] for p in path_small]
    loss_good = [loss[np.argmin(np.abs(w - p))] for p in path_good]
    loss_large = [loss[np.argmin(np.abs(w - p))] for p in path_large]

    plt.plot(path_small[:50], loss_small[:50], 'g-o', markersize=3,
             alpha=0.6, label='Learning Rate 0.01 (too small)')
    plt.plot(path_good, loss_good, 'b-o', markersize=4,
             alpha=0.6, label='Learning Rate 0.05 (good)')
    plt.plot(path_large, loss_large, 'r-o', markersize=4,
             alpha=0.6, label='Learning Rate 0.2 (too large)')

    plt.xlabel('Parameter w')
    plt.ylabel('Loss')
    plt.title('Effect of Learning Rate on Gradient Descent')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Practical Training Visualizations
# =============================================================================

def plot_moons_data(noise=0.3):
    """
    Plot make_moons dataset with given noise level.
    """
    X, y = make_moons(n_samples=200, noise=noise, random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.title(f"Training Data (noise={noise})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


def plot_epoch_experiment(noise=0.3):
    """
    Show train vs test accuracy for different epoch counts.
    """
    X, y = make_moons(n_samples=200, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    epoch_list = [1, 10, 50, 200, 1000, 2000, 5000, 10000]
    train_scores = []
    test_scores = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for max_iter in epoch_list:
            mlp = MLPClassifier(
                hidden_layer_sizes=(40, 40),
                max_iter=max_iter,
                n_iter_no_change=max_iter,
                random_state=42,
            )
            mlp.fit(X_train, y_train)
            train_scores.append(mlp.score(X_train, y_train))
            test_scores.append(mlp.score(X_test, y_test))

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_list, train_scores, "o-", label="Training", linewidth=2)
    plt.plot(epoch_list, test_scores, "s-", label="Test", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_learning_rate_curves(noise=0.1, very_large=False):
    """
    Show loss curves for different learning rates.
    """
    X, y = make_moons(n_samples=200, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    learning_rates = [0.001, 0.01, 0.1]
    if very_large:
        learning_rates.append(1.9)

    plt.figure(figsize=(10, 5))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lr in learning_rates:
            mlp = MLPClassifier(
                hidden_layer_sizes=(40, 40),
                max_iter=100,
                n_iter_no_change=100,
                learning_rate_init=lr,
                solver="sgd",
                random_state=42,
            )
            mlp.fit(X_train, y_train)
            plt.plot(mlp.loss_curve_, label=f"Learning Rate = {lr}", linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Impact of Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_alpha_comparison(noise=0.5):
    """
    Compare different regularization (alpha) values.
    """
    X, y = make_moons(n_samples=200, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    alphas = [0.0, 0.1, 0.5, 1.0, 1.2, 1.5]
    train_accs = []
    test_accs = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for alpha in alphas:
            mlp = MLPClassifier(
                hidden_layer_sizes=(100, 100),
                alpha=alpha,
                max_iter=500,
                random_state=42,
            )
            mlp.fit(X_train, y_train)
            train_accs.append(mlp.score(X_train, y_train))
            test_accs.append(mlp.score(X_test, y_test))

    plt.figure(figsize=(12, 5))

    x = np.arange(len(alphas))
    width = 0.35

    plt.bar(x - width/2, train_accs, width, label='Training', alpha=0.7)
    plt.bar(x + width/2, test_accs, width, label='Test', alpha=0.7)

    plt.xlabel('Alpha (Regularization)')
    plt.ylabel('Accuracy')
    plt.title('Impact of Regularization (alpha)')
    plt.xticks(x, [f'α={a}' for a in alphas])
    plt.ylim([0.5, 1.0])
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_overfitting_example():
    """
    Demonstrate overfitting with complex vs simple model.
    """
    X, y = make_moons(n_samples=50, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42
    )

    model_params = [
        {"name": "Overfit", "hidden": (100,), "alpha": 0.0, "max_iter": 2000, "n_iter_no_change": 2000},
        {"name": "Better", "hidden": (10,), "alpha": 1.0, "max_iter": 500, "n_iter_no_change": 10},
    ]

    results = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for params in model_params:
            mlp = MLPClassifier(
                hidden_layer_sizes=params["hidden"],
                alpha=params["alpha"],
                max_iter=params["max_iter"],
                n_iter_no_change=params["max_iter"],
                random_state=42,
            )
            mlp.fit(X_train, y_train)
            results.append({
                "name": params["name"],
                "label": f"{params['name']}: hidden={params['hidden'][0]}, iters={params['max_iter']}, α={params['alpha']}",
                "train": mlp.score(X_train, y_train),
                "test": mlp.score(X_test, y_test),
            })

    plt.figure(figsize=(10, 5))

    x = np.arange(len(results))
    width = 0.35

    train_scores = [r["train"] for r in results]
    test_scores = [r["test"] for r in results]
    labels = [r["label"] for r in results]


    plt.bar(x - width/2, train_scores, width, label='Training', alpha=0.7)
    plt.bar(x + width/2, test_scores, width, label='Test', alpha=0.7)

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Overfitting: Training vs Test Performance')
    plt.xticks(x, labels)
    plt.ylim([0.0, 1.1])
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    for r in results:
        print(f"{r['name']}: Training={r['train']:.3f}, Test={r['test']:.3f}")
