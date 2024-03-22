import pandas as pd

input_filename = 'front_squat_raw.csv'  # Update with the actual file path
output_filename = 'front_squat_convert.csv'

df = pd.read_csv(input_filename, skiprows=1)  # Skip the first row (header)

sensor_id_suffix = 340
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
