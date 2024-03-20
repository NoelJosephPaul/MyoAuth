import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_emg():
    
    df = pd.read_csv("all_users_filtered_emg_data.csv")

    unique_users = df['Username'].unique()
    for user in unique_users:
        if user == "noel1" or user == "niya1" or user == "nikhil1":
            user_data = df[df['Username'] == user]
            plt.plot(user_data['timestamps'], user_data['emgvalues'], label=user)

    plt.xlabel("Timestamp")
    plt.ylabel("EMG Values")
    plt.title("EMG Values Over Time")
    plt.legend()

    plt.show()

def plot_frequency_domain_not_filtered():

    df = pd.read_csv("all_users_emg_data.csv")

    unique_users = df['Username'].unique()
    for user in unique_users:
        if user == "noel1" or user == "niya1" or user == "nikhil1" or user=="joel1":  # Filter specific users if needed
            user_data = df[df['Username'] == user]
            emg_values = user_data['emgvalues']
            fft_values = np.fft.fft(emg_values)
            frequency_bins = np.fft.fftfreq(len(fft_values))
            plt.plot(frequency_bins, np.abs(fft_values), label=user)

    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain Plot")
    plt.legend()

    
    plt.show()

def plot_frequency_domain_filtered():
    
    df = pd.read_csv("all_users_filtered_emg_data.csv")

    
    unique_users = df['Username'].unique()
    for user in unique_users:
        if user == "noel1" or user == "niya1" or user == "nikhil1" or user=="joel1":  # Filter specific users if needed
            user_data = df[df['Username'] == user]
            emg_values = user_data['emgvalues']
            fft_values = np.fft.fft(emg_values)
            frequency_bins = np.fft.fftfreq(len(fft_values))
            plt.plot(frequency_bins, np.abs(fft_values), label=user)

    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain Plot")
    plt.legend()

    plt.show()

plot_frequency_domain_not_filtered()
plot_frequency_domain_filtered()
#plot_emg()