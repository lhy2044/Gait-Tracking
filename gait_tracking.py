from dataclasses import dataclass
from matplotlib import animation
from scipy.interpolate import interp1d
import imufusion
import matplotlib.pyplot as pyplot
import numpy
import scipy.spatial.transform.rotation as R
import scipy.signal
import pandas as pd

def convert_raw_data_to_csv(input_filename, output_filename, sensor_id_suffix):

    df = pd.read_csv(input_filename, skiprows=1)  # Skip the first row (header)
    df_sensor1 = df[df['sensor'] == sensor_id_suffix]

    # Convert timestamps to relative time in seconds from the first timestamp
    df_sensor1['timestamp'] = df_sensor1['timestamp'] / 1e6
    first_timestamp = df_sensor1['timestamp'].min()
    df_sensor1['Time (s)'] = df_sensor1['timestamp'] - first_timestamp

    # Separate the data into two DataFrames: one for gyroscope and one for accelerometer
    df_gyro = df_sensor1[df_sensor1['sensorType'] == 'GYRO'][['Time (s)', 'x', 'y', 'z']]
    df_acc = df_sensor1[df_sensor1['sensorType'] == 'ACC'][['Time (s)', 'x', 'y', 'z']]
    # make acceleration 10 times smaller and keep everything 7 digits after the decimal point
    df_acc[['x', 'y', 'z']] = df_acc[['x', 'y', 'z']] / 10
    df_acc = df_acc.round(7)

    # Rename the columns for the final DataFrame
    df_gyro.columns = ['Time (s)', 'Gyroscope X (deg/s)', 'Gyroscope Y (deg/s)', 'Gyroscope Z (deg/s)']
    df_acc.columns = ['Time (s)', 'Accelerometer X (g)', 'Accelerometer Y (g)', 'Accelerometer Z (g)']

    # Merge the gyroscope and accelerometer data on 'Time (s)'
    df_final = pd.merge_asof(df_gyro.sort_values('Time (s)'), df_acc.sort_values('Time (s)'), on='Time (s)')

    # Save the processed data to a new CSV file
    df_final.to_csv(output_filename, index=False)


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


# use high pass filter to remove integral drift from velocity
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype="high", analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def euler_and_position(file):
    # Import sensor data
    data = numpy.genfromtxt(file, delimiter=",", skip_header=1)

    sample_rate = 50  # 50 Hz

    timestamp = data[:, 0]
    gyroscope = data[:, 1:4]
    accelerometer = data[:, 4:7]

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


    # Calculate velocity (includes integral drift)
    velocity = numpy.zeros((len(timestamp), 3))

    for index in range(len(timestamp)):
        velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

    velocity[:, 0] = butter_highpass_filter(velocity[:, 0], 0.1, sample_rate)
    velocity[:, 1] = butter_highpass_filter(velocity[:, 1], 0.1, sample_rate)
    velocity[:, 2] = butter_highpass_filter(velocity[:, 2], 0.1, sample_rate)


    # Calculate position
    position = numpy.zeros((len(timestamp), 3))

    for index in range(len(timestamp)):
        position[index] = position[index - 1] + delta_time[index] * velocity[index]


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

def main():
    convert_raw_data_to_csv("front_squat_raw.csv", "front_squat_convert.csv", 1744)
    euler_and_position("front_squat_convert.csv")
    
if __name__ == "__main__":
    main()