import wfdb
import os
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import numpy as np

# Load the ECG signal
record_path = r'C:\Users\drham\OneDrive\Desktop\fyp\ecg preprocessing\100'  # Update path

# Load the ECG signal (100.dat is the record file)
record = wfdb.rdrecord(record_path)

#STEP 1
# Plot the first channel of the ECG signal (first 1000 samples as an example)
plt.plot(record.p_signal[:1000, 0])  # Plot first signal
plt.title("ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()


#STEP 2
# If p_signal is present, proceed with processing
if hasattr(record, 'p_signal'):
    ecg_signal = record.p_signal[:, 0]  # Get the first signal (MLII)
    print(f"ECG Signal Shape: {ecg_signal.shape}")
else:
    print("p_signal not found in the record!")

# Bandpass filter design function
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=360.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Apply bandpass filter to clean the ECG signal
filtered_ecg = bandpass_filter(ecg_signal, lowcut=0.5, highcut=50.0, fs=360.0)

# Plot the original and filtered ECG signals (first 1000 samples for comparison)
plt.figure(figsize=(12, 6))

# Plot original ECG
plt.subplot(2, 1, 1)
plt.plot(ecg_signal[:1000], label='Original ECG')
plt.title("Original ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

# Plot filtered ECG
plt.subplot(2, 1, 2)
plt.plot(filtered_ecg[:1000], label='Filtered ECG', color='orange')
plt.title("Filtered ECG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
# Check if the file exists
if os.path.exists(record_path + '.dat'):
    print("File found!")
else:
    print("File not found. Check the path and file name.")


#STEP 3
# Assuming 'filtered_ecg' is already available and is the filtered ECG signal

# Set the threshold to 50% of the maximum of the filtered signal
threshold = 0.5* max(filtered_ecg)  # Adjust the threshold for better accuracy

# Adjust the minimum distance between peaks (0.6 seconds between R-peaks)
distance = 360 * 0.6  # Minimum 0.6s distance between R-peaks (typical for normal heartbeats)

# Detect R-peaks with the updated parameters
r_peaks, _ = find_peaks(filtered_ecg, height=threshold, distance=distance)

# Plot the filtered ECG signal and mark the R-peaks with red crosses
plt.figure(figsize=(10, 6))
plt.plot(filtered_ecg[:1000], label="Filtered ECG Signal", color='blue')

# Mark R-peaks with red crosses
plt.plot(r_peaks[:1000], filtered_ecg[r_peaks[:1000]], "rx", label="Detected R-peaks")  # Red crosses


plt.title("Filtered ECG Signal with Detected R-peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Print the detected R-peaks indices and their amplitudes for verification
print("Detected R-peaks at indices:", r_peaks[:10])  # Displaying the first 10 detected R-peaks
print("R-peak amplitudes:", filtered_ecg[r_peaks[:10]])  # Show the amplitudes of the first 10 R-peaks


#STEP 4
# Extract patient information from comments
patient_info = record.comments[0]  # First comment holds the information
patient_name = patient_info.split()[2]  # Assuming name is at index 2
patient_age = patient_info.split()[0]  # Assuming age is at index 0
patient_sex = patient_info.split()[1]  # Assuming sex is at index 1

# Bandpass filter design function
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=360.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Apply bandpass filter to clean the ECG signal
filtered_ecg = bandpass_filter(ecg_signal, lowcut=0.5, highcut=50.0, fs=360.0)

# Set the threshold to 50% of the maximum of the filtered signal
threshold = 0.5 * max(filtered_ecg)  # Adjust the threshold for better accuracy

# Adjust the minimum distance between peaks (0.6 seconds between R-peaks)
distance = 360 * 0.6  # Minimum 0.6s distance between R-peaks (typical for normal heartbeats)

# Detect R-peaks with the updated parameters
r_peaks, _ = find_peaks(filtered_ecg, height=threshold, distance=distance)

# Calculate RR intervals (in samples)
rr_intervals = np.diff(r_peaks)

# Plot the RR intervals
plt.figure(figsize=(10, 6))
plt.plot(rr_intervals, marker='o', color='r', linestyle='-', label='RR Intervals (in samples)')
plt.title('RR Intervals Over Time')
plt.xlabel('R-Peak Number')
plt.ylabel('RR Interval (samples)')

# Add patient info to the plot (no classification)
plt.text(0.5, 0.9, f"Patient: {patient_name} (Age: {patient_age}, Sex: {patient_sex})", 
         transform=plt.gca().transAxes, fontsize=12, ha='center')

# Show grid and legend
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print the detected R-peaks indices and their amplitudes for verification
print("Detected R-peaks at indices:", r_peaks[:10])  # Displaying the first 10 detected R-peaks
print("R-peak amplitudes:", filtered_ecg[r_peaks[:10]])  # Show the amplitudes of the first 10 R-peaks

# Output the RR intervals and patient info
print(f"Patient Name: {patient_name}")
print(f"Patient Age: {patient_age}")
print(f"Patient Sex: {patient_sex}")
print("Calculated RR Intervals (in samples):", rr_intervals[:10])  # Print the first 10 RR intervals


#FINAL STEP FOR CLASSIFICATION

# Extract patient information from comments
patient_info = record.comments[0]  # First comment holds the information
patient_name = patient_info.split()[2]  # Assuming name is at index 2
patient_age = patient_info.split()[0]  # Assuming age is at index 0
patient_sex = patient_info.split()[1]  # Assuming sex is at index 1

# Bandpass filter design function
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=360.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Apply bandpass filter to clean the ECG signal
filtered_ecg = bandpass_filter(ecg_signal, lowcut=0.5, highcut=50.0, fs=360.0)

# Set the threshold to 50% of the maximum of the filtered signal
threshold = 0.5 * max(filtered_ecg)  # Adjust the threshold for better accuracy

# Adjust the minimum distance between peaks (0.6 seconds between R-peaks)
distance = 360 * 0.6  # Minimum 0.6s distance between R-peaks (typical for normal heartbeats)

# Detect R-peaks with the updated parameters
r_peaks, _ = find_peaks(filtered_ecg, height=threshold, distance=distance)

# Calculate RR intervals (in samples)
rr_intervals = np.diff(r_peaks)

# Calculate HRV (using SDNN and RMSSD)
sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals
rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))  # RMSSD (root mean square of successive differences)

# Calculate Coefficient of Variation (CV)
mean_rr = np.mean(rr_intervals)
cv = sdnn / mean_rr

# Classification based on CV and RMSSD
if cv > 0.1 or rmssd < 50:  # These thresholds can be adjusted based on your data
    ecg_type = "Arrhythmias ECG"
else:
    ecg_type = "Normal ECG"

# Plot the filtered ECG signal and mark the R-peaks with red crosses
plt.figure(figsize=(10, 6))
plt.plot(filtered_ecg[:1000], label="Filtered ECG Signal", color='blue')

# Mark R-peaks with red crosses
plt.plot(r_peaks[:1000], filtered_ecg[r_peaks[:1000]], "rx", label="Detected R-peaks")  # Red crosses

# Add patient info and ECG classification to the plot
plt.text(0.5, 0.9, f"Patient: {patient_name} (Age: {patient_age}, Sex: {patient_sex})", 
         transform=plt.gca().transAxes, fontsize=12, ha='center')
plt.text(0.5, 0.85, f"ECG Classification: {ecg_type}", 
         transform=plt.gca().transAxes, fontsize=12, ha='center')

plt.title("Filtered ECG Signal with Detected R-peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Print the detected R-peaks indices and their amplitudes for verification
print("Detected R-peaks at indices:", r_peaks[:10])  # Displaying the first 10 detected R-peaks
print("R-peak amplitudes:", filtered_ecg[r_peaks[:10]])  # Show the amplitudes of the first 10 R-peaks

# Output the RR intervals and patient info
print(f"Patient Name: {patient_name}")
print(f"Patient Age: {patient_age}")
print(f"Patient Sex: {patient_sex}")
print("Calculated RR Intervals (in samples):", rr_intervals[:10])  # Print the first 10 RR intervals
print(f"Standard Deviation of RR Intervals: {sdnn}")
print(f"RMSSD: {rmssd}")
print(f"Coefficient of Variation: {cv}")
