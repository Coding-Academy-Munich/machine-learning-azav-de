import matplotlib.pyplot as plt
import numpy as np


def plot_2d_vector(v, title=""):
    plt.figure(figsize=(6, 6))
    plt.quiver(
        0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1, color="blue"
    )
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title or f"2D Vector: {v}")
    plt.show()


def plot_vector_addition(v1, v2):
    v_sum = v1 + v2
    plt.figure(figsize=(8, 6))
    # Main vectors
    plt.quiver(
        0,
        0,
        v1[0],
        v1[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        label="v1",
        alpha=0.7,
    )
    plt.quiver(
        0,
        0,
        v2[0],
        v2[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        label="v2",
        alpha=0.7,
    )
    # Sum vector
    plt.quiver(
        0,
        0,
        v_sum[0],
        v_sum[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        label="v1 + v2",
        linewidth=2,
    )
    # Parallelogram
    plt.quiver(
        v1[0],
        v1[1],
        v2[0],
        v2[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        alpha=0.3,
    )
    plt.quiver(
        v2[0],
        v2[1],
        v1[0],
        v1[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        alpha=0.3,
    )

    plt.xlim(-1, 5)
    plt.ylim(-1, 4)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Vector Addition")
    plt.show()


def plot_dot_product_relationship():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Similar direction
    ax = axes[0]
    v1 = np.array([3, 2])
    v2 = np.array([2, 3])
    ax.quiver(
        0,
        0,
        v1[0],
        v1[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        alpha=0.7,
    )
    ax.quiver(
        0,
        0,
        v2[0],
        v2[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        alpha=0.7,
    )
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Similar direction\nDot product: {np.dot(v1, v2)}")

    # Perpendicular
    ax = axes[1]
    v1 = np.array([3, 0])
    v2 = np.array([0, 3])
    ax.quiver(
        0,
        0,
        v1[0],
        v1[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        alpha=0.7,
    )
    ax.quiver(
        0,
        0,
        v2[0],
        v2[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        alpha=0.7,
    )
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Perpendicular\nDot product: {np.dot(v1, v2)}")

    # Opposite direction
    ax = axes[2]
    v1 = np.array([3, 2])
    v2 = np.array([-2, -3])
    ax.quiver(
        0,
        0,
        v1[0],
        v1[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        alpha=0.7,
    )
    ax.quiver(
        0,
        0,
        v2[0],
        v2[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        alpha=0.7,
    )
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Opposite direction\nDot product: {np.dot(v1, v2)}")

    plt.tight_layout()
    plt.show()


def plot_coefficients_effect(x_in, y_in, coefficients_to_try, intercepts_to_try):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, coef, intercept in zip(axes, coefficients_to_try, intercepts_to_try):
        ax.scatter(x_in, y_in, s=100, alpha=0.7)

        # Prediction line
        x_line = np.linspace(0, 6, 100)
        y_line = coef * x_line + intercept
        ax.plot(x_line, y_line, "r-", label=f"y = {coef}x + {intercept}")

        # Calculate and show errors
        predictions = x_in * coef + intercept
        errors = y_in - predictions.flatten()
        mse = np.mean(errors**2)

        # Draw error lines
        for x, y, pred in zip(x_in.flatten(), y_in, predictions.flatten()):
            ax.plot([x, x], [y, pred], "k--", alpha=0.3)

        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title(f"MSE = {mse:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 14)

    plt.tight_layout()
    plt.show()



# Create a grid of coefficient and intercept values
coef_range = np.linspace(0, 4, 50)
intercept_range = np.linspace(-2, 4, 50)
coef_grid, intercept_grid = np.meshgrid(coef_range, intercept_range)


def compute_mse_grid(X_simple, y_simple):
    mse_grid = np.zeros_like(coef_grid)
    for i in range(len(coef_range)):
        for j in range(len(intercept_range)):
            predictions = X_simple * coef_grid[j, i] + intercept_grid[j, i]
            mse_grid[j, i] = np.mean((y_simple - predictions.flatten()) ** 2)
    return mse_grid


def plot_error_surface(X_simple, y_simple):
    mse_grid = compute_mse_grid(X_simple, y_simple)
    fig = plt.figure(figsize=(12, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(coef_grid, intercept_grid, mse_grid, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("Coefficient")
    ax1.set_ylabel("Intercept")
    ax1.set_zlabel("MSE")
    ax1.set_title("Error Surface (3D)")

    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(coef_grid, intercept_grid, mse_grid, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel("Coefficient")
    ax2.set_ylabel("Intercept")
    ax2.set_title("Error Surface (Contour)")

    # Mark the minimum
    min_idx = np.unravel_index(mse_grid.argmin(), mse_grid.shape)
    best_coef = coef_grid[min_idx]
    best_intercept = intercept_grid[min_idx]
    ax2.plot(best_coef, best_intercept, "r*", markersize=15, label="Minimum")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return best_coef, best_intercept, mse_grid[min_idx]


def plot_gradient_descent_concept():
    """Visualize gradient descent concept."""

    fig, ax = plt.subplots(figsize=(10, 6))
    coef_range = np.linspace(-2, 6, 100)
    errors = [(c - 2) ** 2 for c in coef_range]  # Minimum at coef=2
    ax.plot(coef_range, errors, "b-", linewidth=2)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Error")
    ax.set_title("Gradient Descent Concept")

    steps = [-1, 0, 1, 1.5, 1.8, 1.95, 2.0]
    for i, step in enumerate(steps):
        error = (step - 2) ** 2
        ax.plot(step, error, "ro", markersize=10)
        if i > 0:
            prev_step = steps[i - 1]
            prev_error = (prev_step - 2) ** 2
            ax.arrow(
                prev_step,
                prev_error,
                step - prev_step,
                error - prev_error,
                head_width=0.1,
                head_length=0.05,
                fc="red",
                ec="red",
                alpha=0.5,
            )

    ax.grid(True, alpha=0.3)
    ax.text(2, -0.5, "Minimum", ha="center", fontsize=12)
    plt.show()


def plot_matrix(matrix, title="Matrix Visualization"):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmax=1000)
    feature_names = ["Size", "Beds", "Baths", "Age", "Garage"]
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(feature_names)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"House {i+1}" for i in range(matrix.shape[0])])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(
                j, i, matrix[i, j], ha="center", va="center", color="black"
            )

    ax.set_title(title)
    plt.tight_layout()
    plt.show()
