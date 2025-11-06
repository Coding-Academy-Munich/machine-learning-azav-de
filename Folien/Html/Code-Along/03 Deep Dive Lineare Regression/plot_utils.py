from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_data(X, y, size=200, alpha=0.7):
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, s=size, alpha=alpha)
    plt.xlabel("X (Feature)")
    plt.ylabel("y (Target)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_2d_vector(v, title=""):
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1, color="blue")
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


def plot_coefficients_interactive(x_in, y_in):
    from ipywidgets import interact, FloatSlider

    def update(coef=1.0, intercept=0.0):
        plt.figure(figsize=(8, 6))
        plt.scatter(x_in, y_in, s=100, alpha=0.7)

        # Prediction line
        x_line = np.linspace(0, 6, 100)
        y_line = coef * x_line + intercept
        plt.plot(x_line, y_line, "r-", label=f"y = {coef:.1f}x + {intercept:.1f}")

        # Calculate and show errors
        predictions = x_in * coef + intercept
        errors = y_in - predictions.flatten()
        mse = np.mean(errors**2)

        # Draw error lines
        for x, y, pred in zip(x_in.flatten(), y_in, predictions.flatten()):
            plt.plot([x, x], [y, pred], "k--", alpha=0.3)

        plt.xlabel("X")
        plt.ylabel("y")
        plt.title(f"MSE = {mse:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 6)
        plt.ylim(0, 14)
        plt.show()

    plot = interact(
        update,
        coef=FloatSlider(value=1.0, min=-1.0, max=4.0, step=0.1),
        intercept=FloatSlider(value=0.0, min=0.0, max=5.0, step=0.1),
    )


# Create a grid of coefficient and intercept values
coef_range = np.linspace(1, 3, 101)
intercept_range = np.linspace(-2, 4, 101)
coef_grid, intercept_grid = np.meshgrid(coef_range, intercept_range)


def compute_mse_grid(X_simple, y_simple):
    mse_grid = np.zeros_like(coef_grid)
    for i in range(len(coef_range)):
        for j in range(len(intercept_range)):
            predictions = X_simple * coef_grid[j, i] + intercept_grid[j, i]
            mse_grid[j, i] = np.mean((y_simple - predictions.flatten()) ** 2)
    return mse_grid


def plot_error_surface_3d(X_simple, y_simple, use_log_scale=False, clip_percentile=None, log_epsilon=0.1):
    """Plot interactive 3D error surface (plotly) or static 3D plot (matplotlib)

    Parameters:
    -----------
    X_simple : array-like
        Input features
    y_simple : array-like
        Target values
    use_log_scale : bool, default=True
        If True, uses log scale for MSE to better show curvature
    clip_percentile : float, default=95
        Percentile at which to clip MSE values (helps with extreme values)
    log_epsilon : float, default=0.1
        Small value added before taking log to avoid log(0).
        Use 0.1 for data with min error near 0, use 1e-10 for larger min errors.
    """
    mse_grid = compute_mse_grid(X_simple, y_simple)

    # Find minimum
    min_idx = np.unravel_index(mse_grid.argmin(), mse_grid.shape)
    best_coef = coef_grid[min_idx]
    best_intercept = intercept_grid[min_idx]

    # Apply transformations to improve visualization
    mse_display = mse_grid.copy()
    z_label = "MSE"

    if clip_percentile is not None:
        clip_value = np.percentile(mse_grid, clip_percentile)
        mse_display = np.clip(mse_display, None, clip_value)
        z_label += f" (clipped at {clip_percentile}th percentile)"

    if use_log_scale:
        # Add small epsilon to avoid log(0)
        mse_display = np.log10(mse_display + log_epsilon)
        z_label = f"log₁₀(MSE + {log_epsilon})"

    if PLOTLY_AVAILABLE:
        # Create 3D surface plot with plotly
        # For hover, we need to show the actual MSE value
        # Note: Create text array for hover (more reliable than customdata for Surface)
        hover_text = np.array([[
            f'<b>Coefficient</b>: {coef_grid[i, j]:.3f}<br>'
            f'<b>Intercept</b>: {intercept_grid[i, j]:.3f}<br>'
            f'<b>MSE</b>: {mse_grid[i, j]:.4f}'
            for j in range(coef_grid.shape[1])
        ] for i in range(coef_grid.shape[0])])

        fig = go.Figure(
            data=[
                go.Surface(
                    x=coef_grid,
                    y=intercept_grid,
                    z=mse_display,
                    colorscale="Viridis",
                    name="MSE",
                    text=hover_text,  # Use text instead of customdata (more reliable)
                    hovertemplate='%{text}<extra></extra>',
                )
            ]
        )

        # Update layout with balanced fixed aspect ratio
        # Use simple 1:1:1 ratio for clean, balanced visualization
        fig.update_layout(
            title="Error Surface (3D)",
            scene=dict(
                xaxis_title="Coefficient",
                yaxis_title="Intercept",
                zaxis_title=z_label,
                aspectmode='cube',  # Equal aspect ratio for all axes
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.3)  # Better viewing angle
                )
            ),
            width=700,
            height=600,
        )

        fig.show("notebook")
    else:
        # Fall back to matplotlib 3D plot
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(coef_grid, intercept_grid, mse_display,
                              cmap='viridis', alpha=0.8, edgecolor='none')

        # Mark minimum - use transformed value for plotting
        min_display = mse_display[min_idx]
        ax.scatter([best_coef], [best_intercept], [min_display],
                  color='red', s=200, marker='*',
                  edgecolors='black', linewidths=2,
                  label=f'Min MSE: {mse_grid[min_idx]:.3f}', zorder=5)

        ax.set_xlabel('Coefficient', fontsize=11, labelpad=10)
        ax.set_ylabel('Intercept', fontsize=11, labelpad=10)
        ax.set_zlabel(z_label, fontsize=11, labelpad=10)
        ax.set_title('Error Surface (3D)', fontsize=13, pad=20)

        # Adjust viewing angle for better perspective
        ax.view_init(elev=25, azim=45)

        plt.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
        ax.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.show()

    return best_coef, best_intercept, mse_grid[min_idx]


def plot_error_surface_comparison(X_simple, y_simple, log_epsilon=0.1):
    """Plot side-by-side comparison: original scale vs log scale

    This helps visualize both the overall error landscape and the curvature details.

    Parameters:
    -----------
    X_simple : array-like
        Input features
    y_simple : array-like
        Target values
    log_epsilon : float, default=0.1
        Small value added before taking log to avoid log(0).
        Use 0.1 for data with min error near 0, use 1e-10 for larger min errors.
    """
    from mpl_toolkits.mplot3d import Axes3D

    mse_grid = compute_mse_grid(X_simple, y_simple)

    # Find minimum
    min_idx = np.unravel_index(mse_grid.argmin(), mse_grid.shape)
    best_coef = coef_grid[min_idx]
    best_intercept = intercept_grid[min_idx]

    fig = plt.figure(figsize=(16, 6))

    # Original scale
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(coef_grid, intercept_grid, mse_grid,
                             cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.scatter([best_coef], [best_intercept], [mse_grid[min_idx]],
               color='red', s=200, marker='*',
               edgecolors='black', linewidths=2, zorder=5)
    ax1.set_xlabel('Coefficient', fontsize=10)
    ax1.set_ylabel('Intercept', fontsize=10)
    ax1.set_zlabel('MSE', fontsize=10)
    ax1.set_title('Original Scale\n(Shows overall landscape)', fontsize=11)
    ax1.view_init(elev=25, azim=45)
    plt.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1)

    # Log scale
    ax2 = fig.add_subplot(122, projection='3d')
    mse_log = np.log10(mse_grid + log_epsilon)
    surf2 = ax2.plot_surface(coef_grid, intercept_grid, mse_log,
                             cmap='viridis', alpha=0.8, edgecolor='none')
    ax2.scatter([best_coef], [best_intercept], [mse_log[min_idx]],
               color='red', s=200, marker='*',
               edgecolors='black', linewidths=2, zorder=5)
    ax2.set_xlabel('Coefficient', fontsize=10)
    ax2.set_ylabel('Intercept', fontsize=10)
    ax2.set_zlabel(f'log₁₀(MSE + {log_epsilon})', fontsize=10)
    ax2.set_title('Log Scale\n(Shows curvature detail)', fontsize=11)
    ax2.view_init(elev=25, azim=45)
    plt.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.1)

    plt.suptitle(f'Error Surface Comparison (Min MSE: {mse_grid[min_idx]:.3f})',
                fontsize=13, y=0.98)
    plt.tight_layout()
    plt.show()

    return best_coef, best_intercept, mse_grid[min_idx]
def plot_error_contour(X_simple, y_simple):
    """Plot interactive contour plot of error surface (plotly) or static contour (matplotlib)"""
    mse_grid = compute_mse_grid(X_simple, y_simple)

    # Find minimum
    min_idx = np.unravel_index(mse_grid.argmin(), mse_grid.shape)
    best_coef = coef_grid[min_idx]
    best_intercept = intercept_grid[min_idx]

    if PLOTLY_AVAILABLE:
        # Create contour plot with plotly
        fig = go.Figure()

        fig.add_trace(
            go.Contour(
                x=coef_grid[0],
                y=intercept_grid[:, 0],
                z=mse_grid,
                colorscale="Viridis",
                ncontours=20,
                colorbar=dict(title="MSE"),
            )
        )

        # Mark minimum
        fig.add_trace(
            go.Scatter(
                x=[best_coef],
                y=[best_intercept],
                mode="markers",
                marker=dict(color="red", size=15, symbol="star"),
                name=f"Min ({mse_grid[min_idx]:.3f})",
            )
        )

        # Update layout
        fig.update_layout(
            title="Error Surface (Contour)",
            xaxis_title="Coefficient",
            yaxis_title="Intercept",
            width=700,
            height=600,
        )

        fig.show("notebook")
    else:
        # Fall back to matplotlib contour plot
        plt.figure(figsize=(10, 8))

        contour = plt.contour(coef_grid, intercept_grid, mse_grid,
                             levels=20, cmap='viridis')
        plt.colorbar(contour, label='MSE')

        # Mark minimum
        plt.scatter([best_coef], [best_intercept],
                   color='red', s=200, marker='*',
                   edgecolors='black', linewidths=2,
                   label=f'Min MSE: {mse_grid[min_idx]:.3f}', zorder=5)

        plt.xlabel('Coefficient')
        plt.ylabel('Intercept')
        plt.title('Error Surface (Contour)')
        plt.legend()
        plt.grid(True, alpha=0.3)
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
            text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_geometric_projection(X_geo, y_geo):
    """
    Visualize the geometric interpretation of the normal equation.

    Shows how the prediction Xθ is the orthogonal projection of y onto
    the column space of X, with the error perpendicular to that space.

    Parameters:
    -----------
    X_geo : array-like, shape (n_samples, n_features)
        Design matrix (should include intercept column)
    y_geo : array-like, shape (n_samples,)
        Target vector

    Returns:
    --------
    tuple : (theta, y_pred, error)
        Optimal parameters, predictions, and error vector
    """
    # Solve using normal equation
    theta_geo = np.linalg.inv(X_geo.T @ X_geo) @ X_geo.T @ y_geo

    # Prediction (projection of y onto column space of X)
    y_pred_geo = X_geo @ theta_geo

    # Error (residual)
    error_geo = y_geo - y_pred_geo

    # Column space basis vectors
    col1 = X_geo[:, 0]  # intercept column
    col2 = X_geo[:, 1]  # feature column

    if PLOTLY_AVAILABLE:
        # Create interactive 3D plot with Plotly
        fig = go.Figure()

        origin = [0, 0, 0]

        # Column space basis vectors
        fig.add_trace(go.Scatter3d(
            x=[0, col1[0]], y=[0, col1[1]], z=[0, col1[2]],
            mode='lines+markers',
            line=dict(color='blue', width=6),
            marker=dict(size=4),
            name='Column 1 (intercept)',
            hovertemplate='Column 1<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

        fig.add_trace(go.Scatter3d(
            x=[0, col2[0]], y=[0, col2[1]], z=[0, col2[2]],
            mode='lines+markers',
            line=dict(color='green', width=6),
            marker=dict(size=4),
            name='Column 2 (feature)',
            hovertemplate='Column 2<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

        # Target vector y
        fig.add_trace(go.Scatter3d(
            x=[0, y_geo[0]], y=[0, y_geo[1]], z=[0, y_geo[2]],
            mode='lines+markers',
            line=dict(color='red', width=8),
            marker=dict(size=6),
            name='y (target)',
            hovertemplate='Target y<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

        # Projection Xθ
        fig.add_trace(go.Scatter3d(
            x=[0, y_pred_geo[0]], y=[0, y_pred_geo[1]], z=[0, y_pred_geo[2]],
            mode='lines+markers',
            line=dict(color='purple', width=8),
            marker=dict(size=6),
            name='Xθ (prediction)',
            hovertemplate='Prediction Xθ<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

        # Error vector (from prediction to target)
        fig.add_trace(go.Scatter3d(
            x=[y_pred_geo[0], y_geo[0]],
            y=[y_pred_geo[1], y_geo[1]],
            z=[y_pred_geo[2], y_geo[2]],
            mode='lines+markers',
            line=dict(color='orange', width=8, dash='dash'),
            marker=dict(size=6),
            name='y - Xθ (error)',
            hovertemplate='Error<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))

        # Create mesh for the column space plane
        u = np.linspace(-1, 3, 10)
        v = np.linspace(-1, 3, 10)
        U, V = np.meshgrid(u, v)
        plane_points = U[:, :, np.newaxis] * col1 + V[:, :, np.newaxis] * col2

        fig.add_trace(go.Surface(
            x=plane_points[:, :, 0],
            y=plane_points[:, :, 1],
            z=plane_points[:, :, 2],
            opacity=0.3,
            colorscale=[[0, 'cyan'], [1, 'cyan']],
            showscale=False,
            name='Column space of X',
            hoverinfo='skip'
        ))

        # Update layout with equal aspect ratio to show orthogonality correctly
        fig.update_layout(
            title='Geometric Interpretation: Projection onto Column Space',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3',
                aspectmode='cube',  # Equal aspect ratio on all axes
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=900,
            height=700,
            showlegend=True
        )

        fig.show("notebook")

    else:
        # Fall back to matplotlib 3D plot
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the vectors
        origin = np.zeros(3)

        # Column space basis vectors
        ax.quiver(*origin, *col1, color='blue', alpha=0.6, arrow_length_ratio=0.1,
                 label='Column 1 (intercept)', linewidth=2)
        ax.quiver(*origin, *col2, color='green', alpha=0.6, arrow_length_ratio=0.1,
                 label='Column 2 (feature)', linewidth=2)

        # Target vector y
        ax.quiver(*origin, *y_geo, color='red', alpha=0.8, arrow_length_ratio=0.1,
                 linewidth=3, label='y (target)')

        # Projection Xθ
        ax.quiver(*origin, *y_pred_geo, color='purple', alpha=0.8, arrow_length_ratio=0.1,
                 linewidth=3, label='Xθ (prediction)')

        # Error vector (from origin to error, for visualization purposes)
        # The error vector conceptually exists in the space perpendicular to column space
        ax.quiver(*origin, *error_geo, color='orange', alpha=0.8, arrow_length_ratio=0.1,
                 linewidth=3, label='y - Xθ (error)')

        # Draw the column space (plane spanned by col1 and col2)
        u = np.linspace(-1, 3, 10)
        v = np.linspace(-1, 3, 10)
        U, V = np.meshgrid(u, v)
        plane_points = U[:, :, np.newaxis] * col1 + V[:, :, np.newaxis] * col2

        ax.plot_surface(plane_points[:, :, 0], plane_points[:, :, 1], plane_points[:, :, 2],
                       alpha=0.2, color='cyan')

        # Set equal aspect ratio for proper orthogonality visualization
        max_range = np.array([
            col1.max() - col1.min(),
            col2.max() - col2.min(),
            y_geo.max() - y_geo.min()
        ]).max() / 2.0

        mid_x = (col1.max() + col1.min()) * 0.5
        mid_y = (col2.max() + col2.min()) * 0.5
        mid_z = (y_geo.max() + y_geo.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('Geometric Interpretation: Projection onto Column Space')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return theta_geo, y_pred_geo, error_geo


def plot_matrix_inverse_transformation(v, A):
    """
    Visualize how a matrix transforms a vector and how its inverse undoes it.

    Shows three subplots:
    1. Original vector v
    2. Transformed vector A × v
    3. Vector after applying inverse A⁻¹ × A × v (back to original)

    Parameters:
    -----------
    v : array-like, shape (2,)
        The original 2D vector to transform
    A : array-like, shape (2, 2)
        The transformation matrix
    """
    # Apply transformation
    v_transformed = A @ v

    # Apply inverse (undo transformation)
    A_inv = np.linalg.inv(A)
    v_back = A_inv @ v_transformed

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Original
    axes[0].arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    axes[0].set_xlim(-0.5, 3.5)
    axes[0].set_ylim(-0.5, 3.5)
    axes[0].grid(True)
    axes[0].set_aspect('equal')
    axes[0].set_title('Original Vector v')
    axes[0].axhline(y=0, color='k', linewidth=0.5)
    axes[0].axvline(x=0, color='k', linewidth=0.5)

    # Plot 2: Transformed
    axes[1].arrow(0, 0, v_transformed[0], v_transformed[1],
                  head_width=0.1, head_length=0.1, fc='red', ec='red')
    axes[1].set_xlim(-0.5, 3.5)
    axes[1].set_ylim(-0.5, 3.5)
    axes[1].grid(True)
    axes[1].set_aspect('equal')
    axes[1].set_title('After A: A × v')
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linewidth=0.5)

    # Plot 3: Back to original
    axes[2].arrow(0, 0, v_back[0], v_back[1],
                  head_width=0.1, head_length=0.1, fc='green', ec='green')
    axes[2].set_xlim(-0.5, 3.5)
    axes[2].set_ylim(-0.5, 3.5)
    axes[2].grid(True)
    axes[2].set_aspect('equal')
    axes[2].set_title('After A⁻¹: A⁻¹ × A × v')
    axes[2].axhline(y=0, color='k', linewidth=0.5)
    axes[2].axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # Print verification
    print("Original vector v:")
    print(v)
    print("\nAfter transformation (A × v):")
    print(v_transformed)
    print("\nAfter applying inverse (A⁻¹ × A × v):")
    print(v_back)
    print("\nBack to original?", np.allclose(v, v_back))
