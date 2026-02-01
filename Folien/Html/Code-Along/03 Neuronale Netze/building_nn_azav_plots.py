import numpy as np
import matplotlib.pyplot as plt

# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================================
# Functions from slides_010_limitations_of_linear_models.py
# ============================================================================

def plot_ice_cream_temperature(ice_cream_x, ice_cream_y):
    """Plot ice cream sales vs temperature"""
    plt.scatter([x[0] for x in ice_cream_x], ice_cream_y, alpha=0.6)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Ice Cream Sales")
    plt.title("Linear Relationship")
    plt.show()


def plot_u_shaped_data(x_difficulty, y_reaction):
    """Plot U-shaped reaction time data"""
    plt.scatter(x_difficulty, y_reaction, alpha=0.6)
    plt.xlabel("Task Difficulty")
    plt.ylabel("Reaction Time (seconds)")
    plt.title("U-Shaped Relationship")
    plt.show()


def plot_linear_model_on_u_shaped(x_difficulty, y_reaction, linear_predictions):
    """Plot linear model predictions on U-shaped data"""
    plt.scatter(x_difficulty, y_reaction, alpha=0.6, label="Actual Data")
    plt.plot(x_difficulty, linear_predictions, 'r-', linewidth=2, label="Linear Model")
    plt.xlabel("Task Difficulty")
    plt.ylabel("Reaction Time (seconds)")
    plt.title("Linear Model on Non-Linear Data")
    plt.legend()
    plt.show()


def plot_learning_plateau(x_practice, y_skill):
    """Plot learning plateau curve"""
    plt.scatter(x_practice, y_skill, alpha=0.6)
    plt.xlabel("Practice Hours")
    plt.ylabel("Skill Level")
    plt.title("Learning Plateau")
    plt.show()


def plot_xor_problem(xor_x, xor_y):
    """Plot XOR problem"""
    colors = ['red' if y == 0 else 'blue' for y in xor_y]
    plt.scatter(xor_x[:, 0], xor_x[:, 1], c=colors, s=200, alpha=0.6)
    plt.xlabel("Feature x₁")
    plt.ylabel("Feature x₂")
    plt.title("XOR Problem")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(True)
    plt.show()


def plot_combining_linear_pieces(x_demo, y_piece1, y_piece2, y_combined):
    """Plot combination of linear pieces"""
    plt.plot(x_demo, y_piece1, '--', label="Piece 1", alpha=0.5)
    plt.plot(x_demo, y_piece2, '--', label="Piece 2", alpha=0.5)
    plt.plot(x_demo, y_combined, 'k-', linewidth=2, label="Combined")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Combining Linear Pieces")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# Functions from slides_020_the_neuron.py
# ============================================================================

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)


def plot_sigmoid():
    """Plot sigmoid activation function"""
    x = np.linspace(-10, 10, 200)
    y = sigmoid(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Sigmoid Activation Function")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label="Middle")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axhline(y=1, color='k', linestyle='-', alpha=0.2)
    plt.legend()
    plt.show()


def plot_relu():
    """Plot ReLU activation function"""
    x = np.linspace(-10, 10, 200)
    y = relu(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'g-', linewidth=2)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("ReLU Activation Function")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3, label="Turning point")
    plt.legend()
    plt.show()


def plot_sigmoid_vs_relu():
    """Plot sigmoid vs ReLU comparison"""
    x = np.linspace(-5, 5, 200)
    y_sigmoid = sigmoid(x)
    y_relu = relu(x)
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_sigmoid, 'b-', linewidth=2, label="Sigmoid")
    plt.plot(x, y_relu, 'g-', linewidth=2, label="ReLU")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Sigmoid vs ReLU")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_neuron_response(simple_neuron_func, w1=0.5, w2=-0.3, bias=1.0):
    """Plot interactive 3D neuron response (plotly) or static 3D plot (matplotlib)

    Parameters:
    -----------
    simple_neuron_func : callable
        Function that computes neuron output: (x1, x2, w1, w2, bias) -> (z, output)
    w1, w2, bias : float
        Neuron parameters
    """
    # Create a grid of inputs
    x1_vals = np.linspace(-2, 4, 50)
    x2_vals = np.linspace(-2, 4, 50)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Calculate neuron output for each point
    Z = np.zeros_like(X1)
    for i in range(len(x1_vals)):
        for j in range(len(x2_vals)):
            _, Z[j, i] = simple_neuron_func(X1[j, i], X2[j, i], w1, w2, bias)

    if PLOTLY_AVAILABLE:
        # Create interactive 3D surface plot with plotly
        hover_text = np.array([[
            f'<b>x₁</b>: {X1[i, j]:.2f}<br>'
            f'<b>x₂</b>: {X2[i, j]:.2f}<br>'
            f'<b>Output</b>: {Z[i, j]:.3f}'
            for j in range(X1.shape[1])
        ] for i in range(X1.shape[0])])

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X1,
                    y=X2,
                    z=Z,
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    colorscale='Viridis',
                    colorbar=dict(title="Neuron<br>Output"),
                    name='Neuron Response'
                )
            ]
        )

        fig.update_layout(
            title=f"Neuron Response Surface (w₁={w1}, w₂={w2}, bias={bias})",
            scene=dict(
                xaxis_title="Input x₁",
                yaxis_title="Input x₂",
                zaxis_title="Neuron Output",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=900,
            height=700,
            hovermode='closest'
        )

        fig.show()
    else:
        # Fallback to matplotlib 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.9,
                               edgecolor='none', antialiased=True)

        ax.set_xlabel('Input x₁', fontsize=11)
        ax.set_ylabel('Input x₂', fontsize=11)
        ax.set_zlabel('Neuron Output', fontsize=11)
        ax.set_title(f'Neuron Response Surface (w₁={w1}, w₂={w2}, bias={bias})',
                     fontsize=12, pad=20)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Neuron Output")
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plt.show()


def plot_combining_linear():
    """Show that combining linear functions stays linear"""
    x = np.linspace(-5, 5, 100)
    linear1 = 2 * x + 1
    linear2 = -x + 3
    combined_linear = 0.5 * linear1 + 0.5 * linear2

    plt.figure(figsize=(10, 6))
    plt.plot(x, linear1, '--', alpha=0.5, label="Linear 1")
    plt.plot(x, linear2, '--', alpha=0.5, label="Linear 2")
    plt.plot(x, combined_linear, 'k-', linewidth=2, label="Combined (still linear)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Combining Linear Functions = Still Linear")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_combining_relu():
    """Show that combining with activation becomes non-linear"""
    x = np.linspace(-5, 5, 100)
    activated1 = relu(2 * x + 1)
    activated2 = relu(-x + 3)
    combined_activated = 0.5 * activated1 + 0.5 * activated2

    plt.figure(figsize=(10, 6))
    plt.plot(x, activated1, '--', alpha=0.5, label="ReLU 1")
    plt.plot(x, activated2, '--', alpha=0.5, label="ReLU 2")
    plt.plot(x, combined_activated, 'k-', linewidth=2, label="Combined (non-linear!)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Combining ReLU Functions = Non-Linear")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# Functions from slides_030_building_neural_networks.py
# ============================================================================

def plot_xor_problem_mlp(xor_x, xor_y, figsize=(6, 6)):
    """Visualize the XOR problem"""
    colors = ['red' if y == 0 else 'blue' for y in xor_y]
    plt.figure(figsize=figsize)
    plt.scatter(xor_x[:, 0], xor_x[:, 1], c=colors, s=300, alpha=0.6, edgecolors='black', linewidth=2)
    plt.xlabel("Feature x₁", fontsize=12)
    plt.ylabel("Feature x₂", fontsize=12)
    plt.title("XOR Problem: Can't be separated by a line", fontsize=14)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_circular_data(X_circles, y_circles):
    """Visualize the circular data"""
    colors = ['red' if y == 0 else 'blue' for y in y_circles]
    plt.figure(figsize=(8, 8))
    plt.scatter(X_circles[:, 0], X_circles[:, 1], c=colors, alpha=0.6)
    plt.xlabel("Feature x₁")
    plt.ylabel("Feature x₂")
    plt.title("Circular Data: Linear Model Can't Separate This")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def plot_linear_decision_boundary(xx, yy, Z_linear, X_circles, y_circles):
    """Plot linear model decision boundary"""
    colors = ['red' if y == 0 else 'blue' for y in y_circles]
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z_linear, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])
    plt.scatter(X_circles[:, 0], X_circles[:, 1], c=colors, alpha=0.7, edgecolors='black')
    plt.xlabel("Feature x₁")
    plt.ylabel("Feature x₂")
    plt.title("Linear Model: Poor Separation")
    plt.axis('equal')
    plt.show()


def plot_mlp_decision_boundary(xx, yy, Z_mlp, X_circles, y_circles):
    """Plot MLP decision boundary"""
    colors = ['red' if y == 0 else 'blue' for y in y_circles]
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z_mlp, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])
    plt.scatter(X_circles[:, 0], X_circles[:, 1], c=colors, alpha=0.7, edgecolors='black')
    plt.xlabel("Feature x₁")
    plt.ylabel("Feature x₂")
    plt.title("Neural Network: Excellent Separation!")
    plt.axis('equal')
    plt.show()


# ============================================================================
# Functions from slides_040_training_first_network.py
# ============================================================================

def plot_predictions_comparison(
    student_y_train, linear_train_pred,
    student_y_test, linear_test_pred,
    mlp_train_pred, mlp_test_pred
):
    """Plot comparison of linear regression and neural network predictions"""
    plt.figure(figsize=(14, 6))

    # Linear Regression
    plt.subplot(1, 2, 1)
    plt.scatter(student_y_train, linear_train_pred, alpha=0.5, label="Train")
    plt.scatter(student_y_test, linear_test_pred, alpha=0.5, label="Test")
    plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label="Perfect")
    plt.xlabel("True Final Grade")
    plt.ylabel("Predicted Final Grade")
    plt.title("Linear Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Neural Network
    plt.subplot(1, 2, 2)
    plt.scatter(student_y_train, mlp_train_pred, alpha=0.5, label="Train")
    plt.scatter(student_y_test, mlp_test_pred, alpha=0.5, label="Test")
    plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label="Perfect")
    plt.xlabel("True Final Grade")
    plt.ylabel("Predicted Final Grade")
    plt.title("Neural Network")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ice_cream_comparison(ice_y_test, ice_linear_pred, ice_mlp_pred):
    """Create comparison plot for ice cream predictions"""
    plt.figure(figsize=(14, 6))

    max_val = max(max(ice_y_test), max(ice_linear_pred))

    plt.subplot(1, 2, 1)
    plt.scatter(ice_y_test, ice_linear_pred, alpha=0.6)
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    plt.xlabel("True Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Linear Regression")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(ice_y_test, ice_mlp_pred, alpha=0.6)
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    plt.xlabel("True Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Neural Network")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
