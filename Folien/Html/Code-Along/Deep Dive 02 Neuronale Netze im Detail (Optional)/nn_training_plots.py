import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Functions from slides_010_the_learning_process.py


def plot_training_error(epoch_counts, train_errors, test_errors):
    """Plot how error decreases during training"""
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_counts, train_errors, 'o-', label='Training Error', linewidth=2)
    plt.plot(epoch_counts, test_errors, 's-', label='Test Error', linewidth=2)
    plt.xlabel('Number of Epochs (max_iter)')
    plt.ylabel('Mean Squared Error')
    plt.title('Error Decreases with Training')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_learning_stages(X_train, y_train, x_plot, y_pred_1, y_pred_10, y_pred_100, y_pred_1000):
    """Plot predictions at different training stages"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.flatten()

    # After 1 epoch
    axes[0].scatter(X_train, y_train, alpha=0.6, label='Training Data')
    axes[0].plot(x_plot, y_pred_1, 'r-', linewidth=2, label='Prediction')
    axes[0].set_title('After 1 Epoch')
    axes[0].set_xlabel('Study Hours')
    axes[0].set_ylabel('Grade')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # After 10 epochs
    axes[1].scatter(X_train, y_train, alpha=0.6, label='Training Data')
    axes[1].plot(x_plot, y_pred_10, 'r-', linewidth=2, label='Prediction')
    axes[1].set_title('After 10 Epochs')
    axes[1].set_xlabel('Study Hours')
    axes[1].set_ylabel('Grade')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # After 100 epochs
    axes[2].scatter(X_train, y_train, alpha=0.6, label='Training Data')
    axes[2].plot(x_plot, y_pred_100, 'r-', linewidth=2, label='Prediction')
    axes[2].set_title('After 100 Epochs')
    axes[2].set_xlabel('Study Hours')
    axes[2].set_ylabel('Grade')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # After 1000 epochs
    axes[3].scatter(X_train, y_train, alpha=0.6, label='Training Data')
    axes[3].plot(x_plot, y_pred_1000, 'r-', linewidth=2, label='Prediction')
    axes[3].set_title('After 1000 Epochs')
    axes[3].set_xlabel('Study Hours')
    axes[3].set_ylabel('Grade')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Functions from slides_020_gradient_descent.py


def plot_simple_loss_landscape(w, loss):
    """Plot a simple 1D loss function"""
    plt.figure(figsize=(10, 6))
    plt.plot(w, loss, 'b-', linewidth=2)
    plt.xlabel('Parameter w')
    plt.ylabel('Loss')
    plt.title('Simple Loss Landscape')
    plt.grid(True, alpha=0.3)

    # Mark the minimum
    plt.plot(2, 1, 'r*', markersize=20, label='Minimum (w=2)')
    plt.legend()
    plt.show()


def plot_gradient_descent_steps(w, loss, w_history, loss_history):
    """Plot gradient descent steps on the loss landscape"""
    plt.figure(figsize=(10, 6))
    plt.plot(w, loss, 'b-', linewidth=2, label='Loss Function')

    # Plot the path of gradient descent
    plt.plot(w_history, loss_history, 'ro-', markersize=8, label='Gradient Descent Path')

    # Annotate steps
    plt.plot(w_history[0], loss_history[0], 'g*', markersize=20, label='Start')
    plt.plot(w_history[-1], loss_history[-1], 'r*', markersize=20, label='End')

    plt.xlabel('Parameter w')
    plt.ylabel('Loss')
    plt.title('Gradient Descent: Step by Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_2d_loss_landscape(W1, W2, Loss_2d):
    """Plot 2D loss landscape as a contour plot"""
    plt.figure(figsize=(10, 8))

    contour = plt.contourf(W1, W2, Loss_2d, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Loss')

    # Mark the minimum
    plt.plot(1, -0.5, 'r*', markersize=20, label='Minimum')

    plt.xlabel('Parameter $w_1$')
    plt.ylabel('Parameter $w_2$')
    plt.title('2D Loss Landscape (Contour Plot)')
    plt.legend()
    plt.show()


def plot_2d_loss_landscape_3d(W1, W2, Loss_2d):
    """Plot 2D loss landscape as a 3D surface"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(W1, W2, Loss_2d, cmap='viridis', alpha=0.8)

    # Mark the minimum
    ax.scatter([1], [-0.5], [0.5], color='red', s=200, marker='*', label='Minimum')

    ax.set_xlabel('Parameter w₁')
    ax.set_ylabel('Parameter w₂')
    ax.set_zlabel('Loss')
    ax.set_title('2D Loss Landscape (3D View)')
    plt.colorbar(surf, label='Loss', shrink=0.5)
    ax.legend()
    plt.show()


def plot_local_minimum(w_landscape, loss_with_local):
    """Plot loss landscape with local minimum"""
    plt.figure(figsize=(12, 7))
    plt.plot(w_landscape, loss_with_local, 'b-', linewidth=2)

    # Mark the global minimum
    global_min_idx = np.argmin(loss_with_local)
    plt.plot(w_landscape[global_min_idx], loss_with_local[global_min_idx],
             'g*', markersize=20, label='Global Minimum (Deepest Valley)')

    # Find the local minimum (Funtensee) properly
    # Look for local minima by finding where gradient changes from negative to positive
    # We know it should be around w=-2, so search in the left half
    left_region_mask = w_landscape < 0.5
    left_w = w_landscape[left_region_mask]
    left_loss = loss_with_local[left_region_mask]
    local_min_idx_in_region = np.argmin(left_loss)
    local_min_w = left_w[local_min_idx_in_region]
    local_min_loss = left_loss[local_min_idx_in_region]

    plt.plot(local_min_w, local_min_loss,
             'r*', markersize=20, label='Local Minimum (Funtensee)')

    plt.xlabel('Parameter w')
    plt.ylabel('Loss')
    plt.title('Loss Landscape with Local Minimum')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add annotations
    plt.annotate('Funtensee\n(lokales Minimum / local minimum)',
                 xy=(local_min_w, local_min_loss),
                 xytext=(-2, 5),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))

    plt.annotate('Globales Minimum\nGlobal Minimum',
                 xy=(w_landscape[global_min_idx], loss_with_local[global_min_idx]),
                 xytext=(3.5, 3),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))

    plt.show()


def plot_different_starting_points(w_landscape, loss_with_local, path_left, path_right):
    """Plot gradient descent from different starting points"""
    plt.figure(figsize=(12, 7))
    plt.plot(w_landscape, loss_with_local, 'b-', linewidth=2, label='Loss Landscape')

    # Plot both paths
    loss_left = [loss_with_local[np.argmin(np.abs(w_landscape - w))] for w in path_left]
    loss_right = [loss_with_local[np.argmin(np.abs(w_landscape - w))] for w in path_right]

    plt.plot(path_left, loss_left, 'ro-', markersize=4, alpha=0.6, label='Start Left → Funtensee')
    plt.plot(path_right, loss_right, 'go-', markersize=4, alpha=0.6, label='Start Right → Global Min')

    # Mark starting and ending points
    plt.plot(path_left[0], loss_left[0], 'r^', markersize=15, label='Start Left')
    plt.plot(path_right[0], loss_right[0], 'g^', markersize=15, label='Start Right')

    plt.xlabel('Parameter w')
    plt.ylabel('Loss')
    plt.title('Different Starting Points Lead to Different Minima')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_learning_rates(w_landscape, loss_with_local, path_small, path_good, path_large):
    """Compare different learning rates"""
    plt.figure(figsize=(12, 7))
    plt.plot(w_landscape, loss_with_local, 'b-', linewidth=2, alpha=0.3)

    # Plot paths with different learning rates
    loss_small = [loss_with_local[np.argmin(np.abs(w_landscape - w))] for w in path_small]
    loss_good = [loss_with_local[np.argmin(np.abs(w_landscape - w))] for w in path_good]
    loss_large = [loss_with_local[np.argmin(np.abs(w_landscape - w))] for w in path_large]

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
    plt.show()


# Functions from slides_030_training_in_practice.py


def plot_training_data(X_train, y_train):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color="blue", label="Training Data")
    plt.xlabel("Hours Studied")
    plt.ylabel("Grades")
    plt.title("Training Data: Hours Studied vs. Grades")
    plt.legend()
    plt.show()


def plot_epochs_vs_score(epoch_values, train_scores, test_scores):
    """Plot how R² score improves with more epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_values, train_scores, "o-", label="Training Score", linewidth=2)
    plt.plot(epoch_values, test_scores, "s-", label="Test Score", linewidth=2)
    plt.xlabel("Number of Epochs")
    plt.ylabel("R² Score")
    plt.title("Model Performance vs. Number of Epochs")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_batch_sizes(batch_labels, batch_scores):
    """Compare performance with different batch sizes"""
    plt.figure(figsize=(10, 6))
    plt.bar(batch_labels, batch_scores, color=["skyblue", "lightcoral", "lightgreen"])
    plt.ylabel("Test R² Score")
    plt.title("Impact of Batch Size on Performance")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis="y")
    plt.show()


def plot_learning_rates_sgd_adam(learning_rates, lr_scores_sgd, lr_scores_adam):
    """Compare performance with different learning rates"""
    plt.figure(figsize=(10, 6))
    plt.plot(
        learning_rates, lr_scores_sgd, "o-", markersize=10, linewidth=2, label="SGD"
    )
    plt.plot(
        learning_rates, lr_scores_adam, "o-", markersize=10, linewidth=2, label="Adam"
    )
    plt.xlabel("Learning Rate")
    plt.ylabel("Test R² Score")
    plt.title("Impact of Learning Rate on Performance")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_model_predictions(n_neurons, neuron_counts, X_train, y_train, X_test, y_test,
                           overfit_train_results, overfit_test_results, n_train):
    """Plot model predictions for given number of neurons"""
    index = neuron_counts.index(n_neurons)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:n_train], y_train[:n_train], color="blue", label="Training Data")
    plt.scatter(X_test, y_test, color="purple", label="Test Data")
    plt.scatter(
        X_train[:n_train],
        overfit_train_results[index],
        color="orange",
        label=f"Train Predictions ({n_neurons} Neurons)",
        alpha=0.6,
    )
    plt.scatter(
        X_test,
        overfit_test_results[index],
        color="red",
        label=f"Test Predictions ({n_neurons} Neurons)",
        alpha=0.6,
    )
    plt.xlabel("Hours Studied")
    plt.ylabel("Grades")
    plt.title(f"Model Predictions with {n_neurons} Neurons")
    plt.legend()
    plt.show()


def plot_overfitting(neuron_counts, overfit_train_scores, overfit_test_scores):
    """Show overfitting with increasing model complexity"""
    plt.figure(figsize=(10, 6))
    plt.plot(neuron_counts, overfit_train_scores, 'o-',
             label='Training Score', linewidth=2, markersize=8)
    plt.plot(neuron_counts, overfit_test_scores, 's-',
             label='Test Score', linewidth=2, markersize=8)

    plt.xlabel('Number of Neurons')
    plt.ylabel('R² Score')
    plt.title('Overfitting: Training vs. Test Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Annotate the overfitting region
    plt.axvspan(20, 250, alpha=0.2, color='red', label='Overfitting Region')

    plt.show()


def plot_regularization(alphas, reg_train_scores, reg_test_scores):
    """Show effect of regularization on performance"""
    plt.figure(figsize=(10, 6))
    plt.plot(
        alphas,
        reg_train_scores,
        "o-",
        label="Training Score",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        alphas, reg_test_scores, "s-", label="Test Score", linewidth=2, markersize=8
    )

    plt.xlabel("Alpha (Regularization Strength)")
    plt.ylabel("R² Score")
    plt.title("Effect of Regularization")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Functions from slides_040_comparing_models.py


def plot_linear_comparison(X_train_lin, y_train_lin, x_plot, y_pred_lr, y_pred_mlp):
    """Compare models on linear data"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_lin, y_train_lin, alpha=0.6, label='Training Data')
    plt.plot(x_plot, y_pred_lr, 'r-', linewidth=2, label='Linear Regression')
    plt.plot(x_plot, y_pred_mlp, 'g--', linewidth=2, label='MLP')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Data: Both Models Perform Well')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_nonlinear_comparison(X_train_nl, y_train_nl, x_plot, y_pred_lr_nl, y_pred_mlp_nl):
    """Compare models on non-linear data"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_nl, y_train_nl, alpha=0.6, label='Training Data')
    plt.plot(x_plot, y_pred_lr_nl, 'r-', linewidth=2, label='Linear Regression')
    plt.plot(x_plot, y_pred_mlp_nl, 'g--', linewidth=2, label='MLP')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Non-Linear Data: MLP Performs Better')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_small_data_comparison(X_train_small, y_train_small, X_test_small, y_test_small,
                                x_plot, y_pred_lr_small, y_pred_mlp_small):
    """Compare models on small dataset"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_small, y_train_small, alpha=0.6, s=100, label='Training Data (n=14)')
    plt.scatter(X_test_small, y_test_small, alpha=0.6, s=100, marker='s', label='Test Data (n=6)')
    plt.plot(x_plot, y_pred_lr_small, 'r-', linewidth=2, label='Linear Regression')
    plt.plot(x_plot, y_pred_mlp_small, 'g--', linewidth=2, label='MLP')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Small Dataset: Linear Regression Often More Stable')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_score_comparison(scenarios, lr_test_scores, mlp_test_scores):
    """Compare test scores across scenarios"""
    x_pos = np.arange(len(scenarios))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x_pos - width/2, lr_test_scores, width, label='Linear Regression', alpha=0.8)
    plt.bar(x_pos + width/2, mlp_test_scores, width, label='MLP', alpha=0.8)

    plt.xlabel('Scenario')
    plt.ylabel('Test R² Score')
    plt.title('Model Comparison Across Different Scenarios')
    plt.xticks(x_pos, scenarios)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(-0.5, 1.0)
    plt.show()
