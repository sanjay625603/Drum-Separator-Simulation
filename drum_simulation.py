import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

R = 0.5
omega = 4
g = 9.81
dt = 0.01
steps = 500
num_particles = 50

np.random.seed(1)

positions = np.random.uniform(-R/2, R/2, (num_particles, 2))
velocities = np.zeros((num_particles, 2))

particle_type = np.random.choice([0, 1], size=num_particles)

def update():
    global positions, velocities

    for i in range(num_particles):

        x, y = positions[i]
        r = np.sqrt(x**2 + y**2)

        if r > 0:
            er = np.array([x, y]) / r
            et = np.array([-y, x]) / r

            velocities[i] += (omega**2 * r) * er * dt
            velocities[i] += omega * et * dt

        velocities[i][1] -= g * dt
        positions[i] += velocities[i] * dt

        dist = np.linalg.norm(positions[i])
        if dist >= R:
            normal = positions[i] / dist
            velocities[i] -= 2 * np.dot(velocities[i], normal) * normal
            positions[i] = normal * (R - 0.01)

        # separation
        if particle_type[i] == 0 and positions[i][1] < -0.45:
            positions[i] = np.array([0, 0.3])
            velocities[i] = np.zeros(2)

fig, ax = plt.subplots()
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)
ax.set_aspect('equal')

circle = plt.Circle((0, 0), R, fill=False)
ax.add_patch(circle)

colors = np.where(particle_type == 0, 'blue', 'red')
scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors)

def animate(frame):
    update()
    scat.set_offsets(positions)
    return (scat,)

# ani = animation.FuncAnimation(fig, animate, frames=steps, interval=20)
# plt.show()
# -----------------------------
# EFFICIENCY CALCULATION
# -----------------------------
def run_simulation(omega_value):
    global positions, velocities

    # Reset particles
    positions = np.random.uniform(-R/2, R/2, (num_particles, 2))
    velocities = np.zeros((num_particles, 2))

    separated = 0
    total_small = np.sum(particle_type == 0)

    for _ in range(300):
        for i in range(num_particles):

            x, y = positions[i]
            r = np.sqrt(x**2 + y**2)

            if r > 0:
                er = np.array([x, y]) / r
                et = np.array([-y, x]) / r

                velocities[i] += (omega_value**2 * r) * er * dt
                velocities[i] += omega_value * et * dt

            velocities[i][1] -= g * dt
            positions[i] += velocities[i] * dt

            dist = np.linalg.norm(positions[i])
            if dist >= R:
                normal = positions[i] / dist
                velocities[i] -= 2 * np.dot(velocities[i], normal) * normal
                positions[i] = normal * (R - 0.01)

            # Count separation
            if particle_type[i] == 0 and positions[i][1] < -0.45:
                separated += 1
                positions[i] = np.array([0, 0.3])
                velocities[i] = np.zeros(2)

    efficiency = separated / (total_small * 300)
    return float(efficiency)


# -----------------------------
# TEST DIFFERENT RPMs
# -----------------------------
print("Running efficiency analysis...")

omega_values = np.linspace(1, 8, 8)
efficiencies = []

for w in omega_values:
    eff = run_simulation(w)
    efficiencies.append(eff)

plt.figure()
plt.plot(omega_values, efficiencies, marker='o')
plt.xlabel("Angular Speed (rad/s)")
plt.ylabel("Separation Efficiency")
plt.title("Drum Separator Performance")
plt.grid()
print(efficiencies)
plt.savefig("results/efficiency_plot.png", dpi=300)
print("Graph saved in results folder")




