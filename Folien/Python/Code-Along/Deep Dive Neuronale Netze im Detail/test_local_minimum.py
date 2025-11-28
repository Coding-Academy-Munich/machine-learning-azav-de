"""Test visualization of the corrected local minimum plot"""
import numpy as np
import matplotlib.pyplot as plt

# Create the loss landscape
w_landscape = np.linspace(-5, 5, 200)
valley_local = -2.0 * np.exp(-((w_landscape + 2) ** 2) / 1.0)
valley_global = -3.0 * np.exp(-((w_landscape - 3) ** 2) / 1.5)
loss_with_local = 5.5 + valley_local + valley_global

# Find global minimum
global_min_idx = np.argmin(loss_with_local)

# Find local minimum (left region)
left_region_mask = w_landscape < 0.5
left_w = w_landscape[left_region_mask]
left_loss = loss_with_local[left_region_mask]
local_min_idx_in_region = np.argmin(left_loss)
local_min_w = left_w[local_min_idx_in_region]
local_min_loss = left_loss[local_min_idx_in_region]

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(w_landscape, loss_with_local, 'b-', linewidth=2, label='Loss Landscape')

# Mark both minima
plt.plot(w_landscape[global_min_idx], loss_with_local[global_min_idx],
         'g*', markersize=20, label=f'Global Minimum (w={w_landscape[global_min_idx]:.2f}, loss={loss_with_local[global_min_idx]:.2f})')
plt.plot(local_min_w, local_min_loss,
         'r*', markersize=20, label=f'Local Minimum/Funtensee (w={local_min_w:.2f}, loss={local_min_loss:.2f})')

plt.xlabel('Parameter w', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Corrected Loss Landscape with Clear Local and Global Minima', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add annotations
plt.annotate('Funtensee\n(local minimum)',
             xy=(local_min_w, local_min_loss),
             xytext=(-2, 5.5),
             fontsize=11,
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.annotate('Global Minimum\n(deepest valley)',
             xy=(w_landscape[global_min_idx], loss_with_local[global_min_idx]),
             xytext=(3.5, 4.0),
             fontsize=11,
             arrowprops=dict(arrowstyle='->', color='green', lw=2))

plt.tight_layout()
plt.savefig('local_minimum_corrected.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'local_minimum_corrected.png'")
print(f"\nLocal minimum (Funtensee): w={local_min_w:.2f}, loss={local_min_loss:.2f}")
print(f"Global minimum: w={w_landscape[global_min_idx]:.2f}, loss={loss_with_local[global_min_idx]:.2f}")
print(f"\nThe local minimum is higher than global minimum: {local_min_loss:.2f} > {loss_with_local[global_min_idx]:.2f} ✓")
plt.show()
