from dataclasses import dataclass
from matplotlib import animation
from scipy.interpolate import interp1d
import imufusion
import matplotlib.pyplot as pyplot
import numpy
import scipy.spatial.transform.rotation as R
import scipy.signal

def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert angles from degrees to radians
    roll = numpy.radians(roll)
    pitch = numpy.radians(pitch)
    yaw = numpy.radians(yaw)
    
    # Calculate rotation matrices
    Rx = numpy.array([[1, 0, 0],
                   [0, numpy.cos(roll), -numpy.sin(roll)],
                   [0, numpy.sin(roll), numpy.cos(roll)]])
    
    Ry = numpy.array([[numpy.cos(pitch), 0, numpy.sin(pitch)],
                   [0, 1, 0],
                   [-numpy.sin(pitch), 0, numpy.cos(pitch)]])
    
    Rz = numpy.array([[numpy.cos(yaw), -numpy.sin(yaw), 0],
                   [numpy.sin(yaw), numpy.cos(yaw), 0],
                   [0, 0, 1]])
    
    # Combine the rotations
    R = numpy.dot(Rz, numpy.dot(Ry, Rx))
    return R

# Import sensor data ("short_walk.csv" or "long_walk.csv")
data = numpy.genfromtxt("front_squat_convert.csv", delimiter=",", skip_header=1)

sample_rate = 50  # 50 Hz

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]

# Plot sensor data
figure, axes = pyplot.subplots(nrows=6, sharex=True, gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 1]})

figure.suptitle("Sensors data, Euler angles, and AHRS internal states")

axes[0].plot(timestamp, gyroscope[:, 0], "tab:red", label="Gyroscope X")
axes[0].plot(timestamp, gyroscope[:, 1], "tab:green", label="Gyroscope Y")
axes[0].plot(timestamp, gyroscope[:, 2], "tab:blue", label="Gyroscope Z")
axes[0].set_ylabel("Degrees/s")
axes[0].grid()
axes[0].legend()

axes[1].plot(timestamp, accelerometer[:, 0], "tab:red", label="Accelerometer X")
axes[1].plot(timestamp, accelerometer[:, 1], "tab:green", label="Accelerometer Y")
axes[1].plot(timestamp, accelerometer[:, 2], "tab:blue", label="Accelerometer Z")
axes[1].set_ylabel("g")
axes[1].grid()
axes[1].legend()

# Instantiate AHRS algorithms
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,
                                   0.5,  # gain
                                   2000,  # gyroscope range
                                   10,  # acceleration rejection
                                   0,  # magnetic rejection
                                   5 * sample_rate)  # rejection timeout = 5 seconds

# Process sensor data
delta_time = numpy.diff(timestamp, prepend=timestamp[0])

euler = numpy.empty((len(timestamp), 3))
internal_states = numpy.empty((len(timestamp), 3))
acceleration = numpy.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])

    ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])

    euler[index] = ahrs.quaternion.to_euler()

    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = numpy.array([ahrs_internal_states.acceleration_error,
                                          ahrs_internal_states.accelerometer_ignored,
                                          ahrs_internal_states.acceleration_recovery_trigger])

    acceleration[index] = 9.81 * ahrs.earth_acceleration  # convert g to m/s/s

# Plot Euler angles
axes[2].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
axes[2].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
axes[2].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
axes[2].set_ylabel("Degrees")
axes[2].grid()
axes[2].legend()

# Plot internal states
axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
axes[3].set_ylabel("Degrees")
axes[3].grid()
axes[3].legend()

axes[4].plot(timestamp, internal_states[:, 1], "tab:cyan", label="Accelerometer ignored")
pyplot.sca(axes[4])
pyplot.yticks([0, 1], ["False", "True"])
axes[4].grid()
axes[4].legend()

axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration recovery trigger")
axes[5].set_xlabel("Seconds")
axes[5].grid()
axes[5].legend()

pyplot.savefig("plot1.png")

# Plot acceleration
_, axes = pyplot.subplots(nrows=4, sharex=True, gridspec_kw={"height_ratios": [6, 1, 6, 6]})

axes[0].plot(timestamp, acceleration[:, 0], "tab:red", label="X")
axes[0].plot(timestamp, acceleration[:, 1], "tab:green", label="Y")
axes[0].plot(timestamp, acceleration[:, 2], "tab:blue", label="Z")
axes[0].set_title("Acceleration")
axes[0].set_ylabel("m/s/s")
axes[0].grid()
axes[0].legend()


# Calculate velocity (includes integral drift)
velocity = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

# use high pass filter to remove integral drift from velocity
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype="high", analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y

velocity[:, 0] = butter_highpass_filter(velocity[:, 0], 0.1, sample_rate)
velocity[:, 1] = butter_highpass_filter(velocity[:, 1], 0.1, sample_rate)
velocity[:, 2] = butter_highpass_filter(velocity[:, 2], 0.1, sample_rate)

# Plot velocity
axes[2].plot(timestamp, velocity[:, 0], "tab:red", label="X")
axes[2].plot(timestamp, velocity[:, 1], "tab:green", label="Y")
axes[2].plot(timestamp, velocity[:, 2], "tab:blue", label="Z")
axes[2].set_title("Velocity")
axes[2].set_ylabel("m/s")
axes[2].grid()
axes[2].legend()

# Calculate position
position = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    position[index] = position[index - 1] + delta_time[index] * velocity[index]

# Plot position
axes[3].plot(timestamp, position[:, 0], "tab:red", label="X")
axes[3].plot(timestamp, position[:, 1], "tab:green", label="Y")
axes[3].plot(timestamp, position[:, 2], "tab:blue", label="Z")
axes[3].set_title("Position")
axes[3].set_xlabel("Seconds")
axes[3].set_ylabel("m")
axes[3].grid()
axes[3].legend()

# write the plot to a file
pyplot.savefig("plot.png")


# Print error as distance between start and final positions
print("Error: " + "{:.3f}".format(numpy.sqrt(position[-1].dot(position[-1]))) + " m")

# Create 3D animation (takes a long time, set to False to skip)
if True:
    figure = pyplot.figure(figsize=(10, 10))

    axes = pyplot.axes(projection="3d")
    axes.set_xlabel("m")
    axes.set_ylabel("m")
    axes.set_zlabel("m")

    x = []
    y = []
    z = []

    scatter = axes.scatter(x, y, z)

    fps = 10
    samples_per_frame = int(sample_rate / fps)

    # Calculate global minima and maxima for x, y, z
    x_min, x_max = numpy.min(position[:, 0]), numpy.max(position[:, 0])
    y_min, y_max = numpy.min(position[:, 1]), numpy.max(position[:, 1])
    z_min, z_max = numpy.min(position[:, 2]), numpy.max(position[:, 2])

    # Assuming your axes aspect ratio setup is necessary for the visual, but you can adjust it to fit the fixed scope better if needed
    axes.set_box_aspect((numpy.ptp(position[:, 0]), numpy.ptp(position[:, 1]), numpy.ptp(position[:, 2])))

    # Update function modification: remove dynamic scope adjustment
    def update(frame):
        index = frame * samples_per_frame
        axes.set_title("{:.3f}".format(timestamp[index]) + " s")

        # Update to only show the current position
        x_current = position[index, 0]
        y_current = position[index, 1]
        z_current = position[index, 2]

        # Clear previous scatter plot
        axes.clear()
        
        # Set fixed scope for the animation
        axes.set_xlim3d(x_min, x_max)
        axes.set_ylim3d(y_min, y_max)
        axes.set_zlim3d(z_min, z_max)

        # Create a new scatter plot with just the current position
        scatter = axes.scatter([x_current], [y_current], [z_current], c='blue')

        # Clear previous orientation lines
        while len(axes.lines) > 0:
            axes.lines[0].remove()

        # Current orientation represented as a rotation matrix
        rotation_matrix = euler_to_rotation_matrix(euler[index, 0], euler[index, 1], euler[index, 2])

        # Origin of the orientation vectors
        origin = position[index, :]

        # Draw new orientation lines for the current frame
        for axis, color in zip(rotation_matrix, ['r', 'g', 'b']):
            axes.quiver(*origin, *axis, length=0.1, color=color)

        print("Frame: " + str(frame) + " / " + str(int(len(timestamp) / samples_per_frame)))
        return scatter

    anim = animation.FuncAnimation(figure, update,
                                frames=int(len(timestamp) / samples_per_frame),
                                interval=1000 / fps,
                                repeat=False)

    anim.save("animation.gif", writer=animation.PillowWriter(fps))

pyplot.show()
