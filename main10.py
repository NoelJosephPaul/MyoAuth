#main9.py de 2nd version...here knn is replaced by CNN model tto

#main8.py de 2 second version

#main5 cont and separate login and register

#introducing a new title window and audio

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import serial
import threading
from datetime import datetime
import time
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
from PIL import Image, ImageTk  # Import Image and ImageTk from PIL
import pygame  # Import pygame library
import imageio
from ttkthemes import ThemedStyle
from tkinter import font
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import os


class EMGRecorderApp:
    def __init__(self, root, serial_port):
        self.root = root
        self.root.title("MyoAuth")
        self.root.geometry("700x500")  # Set window size
        self.root.iconbitmap(default='myoauthicon.ico')
        self.root.configure(bg='white')

        style = ThemedStyle(self.root)
        style.set_theme("plastik") 

        self.gif_label = None
        self.step = None
        self.user_name = None
        self.initial_option = None

        self.cnn_model = None
        self.scaler = None

        # Load the CNN model
        self.load_cnn_model()

        # Load and display the image
        image_path = 'myoauthicon.png'  # Replace with your image path
        original_image = Image.open(image_path)
        resized_image = original_image.resize((200, 200), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(resized_image)
        self.image_label = tk.Label(self.root, image=self.image,background="white")
        self.image_label.place(relx=0.5, rely=0.4, anchor='center')

        # Create a label for the title
        self.title_label = ttk.Label(self.root, text="MyoAuth", font=("Calibri", 25), foreground='green',background="white")
        self.title_label.place(relx=0.5, rely=0.65, anchor='center')

        # Play audio when the title is shown
        pygame.init()
        pygame.mixer.music.load('myoauthintro.mp3')  # Replace with your audio file path
        pygame.mixer.music.play(0)  # Start playing
        self.fade_out_audio(1000, 0.1)  # Fade in over 3000 milliseconds (3 seconds) with a step of 0.5

        # Schedule the transition to the main window after 5000 milliseconds (5 seconds)
        self.root.after(2500, self.show_main_window)

    def load_cnn_model(self):
        model_filename = "cnn_model.h5"
        if os.path.exists(model_filename):
            self.cnn_model = models.load_model(model_filename)
            print(f"Trained CNN model loaded from {model_filename}")
        else:
            print("No trained CNN model found. Train the model first.")
    
    def load_cnn_scaler(self):
        # Load the trained scaler for CNN
        cnn_scaler_filename = "cnn_scaler.joblib"
        if os.path.exists(cnn_scaler_filename):
            self.scaler = joblib.load(cnn_scaler_filename)
            print(f"Trained CNN scaler loaded from {cnn_scaler_filename}")
        else:
            print("No trained CNN scaler found.")

    def fade_out_audio(self, duration, step):
        current_volume = 2.0
        # Gradually decrease the volume to simulate a fade-out effect
        for i in range(int(duration / step)):
            current_volume -= step
            pygame.mixer.music.set_volume(max(0.1, current_volume))
            pygame.time.delay(int(step))

    def show_main_window(self):
        # Destroy the title label
        self.title_label.destroy()
        self.image_label.destroy()

        self.clear_window()

        # Create buttons for the main window
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")

        welcome_label = tk.Label(self.root, text="Welcome to MyoAuth", background="white", font=("Helvetica", 15, "bold"), fg="green")
        welcome_label.place(relx=0.5, rely=0.1, anchor='center')

        existing_user_label = tk.Label(self.root, text="Existing User?",background="white", font=("Helvetica", 10))
        existing_user_label.place(relx=0.5, rely=0.3, anchor='center')

        # Create "Login" and "Register" buttons
        self.login_button = ttk.Button(self.root, text="Login", command=self.show_login_window)
        self.login_button.place(relx=0.5, rely=0.37, anchor='center', width=150, height=40)

        new_user_label = tk.Label(self.root, text="New User?",background="white", font=("Helvetica", 10))
        new_user_label.place(relx=0.5, rely=0.6, anchor='center')

        self.register_button = ttk.Button(self.root, text="Register", command=self.show_register_window)
        self.register_button.place(relx=0.5, rely=0.67, anchor='center', width=150, height=40)


        self.serial_port = serial_port
        self.emg_data = {}
        self.data_thread = None


    def show_login_window(self):
        # Clear the window
        self.clear_window()

        self.initial_option = "login"

        login_label = tk.Label(self.root, text="User Login", background="white", font=("Helvetica", 15, "bold"), fg="green")
        login_label.place(relx=0.5, rely=0.1, anchor='center')

        self.user_label = ttk.Label(self.root, text="Enter Username", background="white", font=("Helvetica", 12))
        self.user_label.place(relx=0.5, rely=0.35, anchor='center')

        self.user_entry = ttk.Entry(self.root, justify="center", font=("Helvetica", 12))
        self.user_entry.place(relx=0.5, rely=0.42, anchor='center', width=200)

        self.verify_button = ttk.Button(self.root, text="Verify", command=self.verify_person)
        self.verify_button.place(relx=0.5, rely=0.55, anchor='center', width=150, height=35)

        self.back_button = ttk.Button(self.root, text="Back", command=self.show_main_window)
        self.back_button.place(relx=0.5, rely=0.65, anchor='center', width=150, height=35)

    def show_register_window(self):
        # Clear the window
        self.clear_window()

        self.initial_option = "register"

        register_label = tk.Label(self.root, text="New User Registration", background="white", font=("Helvetica", 15, "bold"), fg="green")
        register_label.place(relx=0.5, rely=0.1, anchor='center')
        
        self.step = 0
        
        # Create "Record EMG Signals," "Extract Features," and "Train" buttons
        self.record_button = ttk.Button(self.root, text="Record EMG Signals", command=self.start_recording_instructions)
        self.record_button.place(relx=0.5, rely=0.3, anchor='center', width=200, height=40)

        self.preprocess_button = ttk.Button(self.root, text="Extract Features", command=self.feature_extract_data)
        self.preprocess_button.place(relx=0.5, rely=0.4, anchor='center', width=200, height=40)

        self.train_button = ttk.Button(self.root, text="Train", command=self.train_classifier)
        self.train_button.place(relx=0.5, rely=0.5, anchor='center', width=150, height=35)

        self.back_button = ttk.Button(self.root, text="Back", command=self.show_main_window)
        self.back_button.place(relx=0.5, rely=0.6, anchor='center', width=150, height=35)

    def step_counter(self):
        # Add your logic for the step counter here
        # This function will be called when the "Next" button is clicked
        self.step = self.step + 1
        if(self.step<=5):
            self.start_recording()
        else:
            self.show_register_window()

    def start_recording_instructions(self):
        
        self.clear_window()
        self.step=0

        # Create instructions label
        self.instructions_label = ttk.Label(self.root, text="Follow these instructions", font=("Helvetica", 14,"bold"), background="white",foreground="green")
        self.instructions_label.place(relx=0.5, rely=0.1, anchor='center')
        

        self.instructions_label2 = ttk.Label(self.root, text="You will be guided through a series of five steps.", font=("Helvetica", 12), background="white")
        self.instructions_label2.place(relx=0.5, rely=0.2, anchor='center')

        self.instructions_label3 = ttk.Label(self.root, text="During each step, execute a hand gesture for a duration of 2 seconds.", font=("Helvetica", 12), background="white")
        self.instructions_label3.place(relx=0.5, rely=0.3, anchor='center')

        self.instructions_label4 = ttk.Label(self.root, text="It is necessary to maintain uniformity by consistently performing the same gesture at each step.", font=("Helvetica", 12), background="white")
        self.instructions_label4.place(relx=0.5, rely=0.35, anchor='center')

        self.user_label = ttk.Label(self.root, text="Before we start, enter Username", font=("Helvetica", 11), background="white")
        self.user_label.place(relx=0.5, rely=0.45, anchor='center')

        self.user_entry = ttk.Entry(self.root, justify="center", font=("Helvetica", 12),width=30)
        self.user_entry.place(relx=0.5, rely=0.52, anchor='center')

        # Create "Start Recording" button
        self.next_button = ttk.Button(self.root, text="Start Recording", command=self.step_counter)
        self.next_button.place(relx=0.5, rely=0.6, anchor='center')

        self.back_button = ttk.Button(self.root, text="Back", command=self.show_register_window)
        self.back_button.place(relx=0.5, rely=0.7, anchor='center')

    def clear_window(self):
        # Destroy all widgets in the window
        for widget in self.root.winfo_children():
            widget.destroy()

    def start_recording(self):
        if(self.step==1):
            user_name = self.user_entry.get()
            self.user_name = user_name
            if(user_name!=""):
                user_name=user_name+str(self.step)
        else:
            user_name=self.user_name+str(self.step)
        
        print(user_name)
            
        if not user_name:
            tk.messagebox.showinfo("Error", "Please enter Username.")
            self.step=0
            return

        if user_name not in self.emg_data:
            self.emg_data[user_name] = []  # Create an empty list for the user if not present

        self.emg_data.clear()
        self.emg_data[user_name] = []
        self.clear_window()

        self.data_thread = threading.Thread(target=self.record_emg, args=(user_name,))
        self.data_thread.start()

        step_label = tk.Label(self.root, text="", font=("Helvetica", 15,"bold"), background="white",foreground="green")
        step_label.place(relx=0.5, rely=0.1, anchor='center')
        step_label.config(text=f"Step {self.step} / 5")

        countdown_label = tk.Label(self.root, text="", font=("Helvetica", 12), background="white")
        countdown_label.place(relx=0.5, rely=0.2, anchor='center')

        # Start the countdown in a separate thread
        countdown_thread = threading.Thread(target=self.start_countdown, args=(2, countdown_label))
        countdown_thread.start()
    
    def start_countdown(self, seconds, countdown_label):

        gif_path = 'wave2.gif'  # Replace with your GIF path
        self.gif_label = tk.Label(self.root,background="white")
        self.gif_label.place(relx=0.5, rely=0.3, anchor='center')
        self.show_gif(gif_path, 2)  # Display the GIF for 5 seconds

        for i in range(0,seconds,1):
            countdown_label.config(text=f"Recording...  {i} seconds")
            time.sleep(1)
        countdown_label.config(text="Recording Complete")

    def show_gif(self, gif_path, duration):
        # Create a label to display the GIF
        self.gif_label = tk.Label(self.root,background="white")
        self.gif_label.place(relx=0.5, rely=0.4, anchor='center')

        # Load the animated GIF
        gif = imageio.mimread(gif_path)
        gif_images = [Image.fromarray(img) for img in gif]

        # Convert the PIL Images to Tkinter-compatible format
        self.gif_photos = [ImageTk.PhotoImage(img) for img in gif_images]

        # Display the animated GIF
        self.animate_gif(0, duration)

    def animate_gif(self, index, duration):
        # Display the current frame of the animated GIF
        self.gif_label.config(image=self.gif_photos[index])

        # Schedule the next frame to be shown after a delay
        if index < len(self.gif_photos) - 1:
            self.root.after(int(duration * 1000 / len(self.gif_photos)), self.animate_gif, index + 1, duration)
        elif self.initial_option=="register":
            # Schedule the static image to be shown after the animated GIF
            self.root.after(int(duration * 160), self.show_static_image, 'tick2.png')
        else:
            pass

    def show_static_image(self, image_path):
        static_image = Image.open(image_path)
        resized_image = static_image.resize((100, 100), Image.LANCZOS)
        static_photo = ImageTk.PhotoImage(resized_image)

        # Replace the GIF label with a static image label
        self.gif_label.config(image=static_photo)
        self.gif_label.image = static_photo

        # Play audio
        pygame.init()
        pygame.mixer.music.load('myoauthintro.mp3')  # Replace with your audio file path
        pygame.mixer.music.play(0,0,1)  # Start playing
        
        # Create "Next" button
        if(self.step<5):
            self.next_button = ttk.Button(self.root, text="Next Step", command=self.step_counter)
            self.next_button.place(relx=0.5, rely=0.6, anchor='center')
        else:
            self.next_button = ttk.Button(self.root, text="Finish", command=self.step_counter)
            self.next_button.place(relx=0.5, rely=0.6, anchor='center')


    def record_emg(self, user_name):
        print(user_name)
        with serial.Serial(self.serial_port, 115200, timeout=1) as ser:
            ser.reset_input_buffer()

            start_time = time.time()
            while time.time() - start_time < 2:
                try:
                    line = ser.readline().decode().strip()
                    if line:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        emg_value = float(line)
                        self.emg_data[user_name].append((timestamp, emg_value))
                except ValueError as e:
                    print(f"Error parsing data: {e}")

        # Save data to CSV after recording
        self.save_to_csv()

    def save_to_csv(self):
        filename = "all_users_emg_data.csv"

        # Check if the file already exists
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header only if the file doesn't exist
            if not file_exists:
                csv_writer.writerow(["Username", "timestamps", "emgvalues"])

            for user_name, data in self.emg_data.items():
                print(user_name)
                for timestamp, emg_value in data:
                    csv_writer.writerow([user_name, timestamp, emg_value])

        print(f"EMG data saved to {filename}")

    def feature_extract_data(self):
        input_file = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

        if not input_file:
            return

        output_file = filedialog.asksaveasfilename(title="Save Feature data as", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

        if not output_file:
            return

        self.extract_features(input_file, output_file)

    def extract_features(self, input_file, output_file):
        df = pd.read_csv(input_file)

        feature_vectors = []
        for user_name, user_data in df.groupby('Username'):
            emg_values = user_data['emgvalues'].tolist()
            features = self.calculate_features(emg_values)
            user_name = ''.join(char for char in user_name if not char.isdigit())
            feature_vectors.append({'Username': user_name, 'Features': features})

        feature_df = pd.DataFrame(feature_vectors)
        feature_df.to_csv(output_file, index=False)

        print(f"\nFeature vectors saved to {output_file}")

    def calculate_features(self, emg_values):
        # feature extraction
        features = []
        features.append(np.mean(np.abs(emg_values),axis=0)) #mean absolute value
        features.append(np.sum(np.abs(np.diff(emg_values)),axis=0)) #waveform length
        features.append(np.sum(np.diff(np.sign(emg_values),axis=0)!=0,axis=0)/(len(emg_values)-1))
        features.append(skew(emg_values,axis=0))
        features.append(kurtosis(emg_values,axis=0))
        features.append(np.sqrt(np.mean(np.array(emg_values)**2,axis=0))) #root mean sqaure
        features.append(np.sum(np.array(emg_values)**2,axis=0)) #simple square integral

        # Add Fourier transform as a new feature
        fourier_transform = np.abs(np.fft.fft(emg_values))
        features.append(np.mean(fourier_transform, axis=0))

        # Additional frequency domain features
        features.append(np.sum(fourier_transform, axis=0))  # Total energy
        features.append(np.sum(fourier_transform ** 2, axis=0))  # Power
        features.append(np.argmax(fourier_transform, axis=0))  # Dominant frequency index

        # Frequency centroid
        frequency_bins = len(fourier_transform)
        frequency_values = np.fft.fftfreq(frequency_bins, d=1.0)  # Frequency values corresponding to bins
        features.append(np.sum(frequency_values * fourier_transform, axis=0) / np.sum(fourier_transform, axis=0))

        # Spectral entropy
        spectral_entropy = -np.sum(fourier_transform * np.log2(fourier_transform + 1e-10), axis=0)
        features.append(spectral_entropy)

        return features
    
    def train_classifier(self):
        # Load feature vectors from the CSV file
        input_file = filedialog.askopenfilename(title="Select Feature Vectors CSV file", filetypes=[("CSV files", "*.csv")])

        if not input_file:
            return

        print("Loading feature vectors...")
        df = pd.read_csv(input_file)
        # Group by username and split the data
        train_df, test_df = [], []
        for username, group in df.groupby('Username'):
            # Select 4 random samples for training
            train_data = group.sample(n=4, random_state=42)
            # Remaining one sample for testing
            test_data = group.drop(train_data.index)

            train_df.append(train_data)
            test_df.append(test_data)

        # Combine the dataframes for training and testing
        train_df = pd.concat(train_df)
        test_df = pd.concat(test_df)

        # Extract features and labels
        X_train = np.array([eval(features) for features in train_df['Features']])
        y_train = train_df['Username'].astype('category').cat.codes.values

        X_test = np.array([eval(features) for features in test_df['Features']])
        y_test = test_df['Username'].astype('category').cat.codes.values

        # Standardize the input features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Reshape the data for CNN input
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build the CNN model
        self.cnn_model = Sequential()
        self.cnn_model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        self.cnn_model.add(MaxPooling1D(pool_size=2))
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dense(128, activation='relu'))
        self.cnn_model.add(Dense(len(np.unique(y_train)), activation='softmax'))

        # Compile the model
        self.cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        self.cnn_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

        # Save the trained model
        self.cnn_model.save("cnn_model.h5")
        joblib.dump(scaler, "cnn_scaler.joblib")

        print("CNN Classifier Trained Successfully!")

    def verify_person(self):

        '''if not self.cnn_classifier:
            messagebox.showinfo("Error", "Please train the classifier first.")
            return'''

        user_name = self.user_entry.get()

        if not user_name:
            tk.messagebox.showinfo("Error", "Please enter Username.")
            return
        
        self.clear_window()

        self.data_thread = threading.Thread(target=self.record_emg_for_verification, args=(user_name,))
        self.data_thread.start()

        countup_label = tk.Label(self.root, text="", font=("Helvetica", 12), background="white")
        countup_label.place(relx=0.5, rely=0.2, anchor='center')

        self.verify_countup = threading.Thread(target=self.verify_countdown, args=(2,countup_label))
        self.verify_countup.start()

    def verify_countdown(self, seconds, countup_label):
        gif_path = 'wave2.gif'  # Replace with your GIF path
        self.gif_label = tk.Label(self.root,background="white")
        self.gif_label.place(relx=0.5, rely=0.3, anchor='center')
        self.show_gif(gif_path, 2)  # Display the GIF for 5 seconds

        for i in range(0,seconds,1):
            countup_label.config(text=f"Recording...  {i} seconds")
            time.sleep(1)
        
        countup_label.destroy()
        self.gif_label.destroy()

    def record_emg_for_verification(self, user_name):
        with serial.Serial(self.serial_port, 115200, timeout=1) as ser:
            ser.reset_input_buffer()

            start_time = time.time()
            emg_values = []

            while time.time() - start_time < 2:  # Record for 5 seconds for verification
                try:
                    line = ser.readline().decode().strip()
                    if line:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        emg_value = float(line)
                        emg_values.append(emg_value)
                except ValueError as e:
                    print(f"Error parsing data: {e}")

        # Convert EMG signals to feature vector
        features = self.calculate_features(emg_values)
        #print(features)

        # Standardize the feature values (skip this step for CNN)
        scaler=joblib.load("cnn_scaler.joblib")
        features_scaled = scaler.transform([features])

        # Reshape the data for CNN input
        #features_reshaped = np.array(features_scaled).reshape(1, len(features_scaled), 1)

        self.load_cnn_model()
        self.load_cnn_scaler()
        # Load the trained CNN model
        model = load_model("cnn_model.h5")

        # Predict the person using the trained CNN model
        predictions = model.predict(features_scaled)
        print(predictions)
        predicted_label = np.argmax(predictions)
        print(predicted_label)

        self.clear_window()

        self.verification_label = ttk.Label(self.root, text="Verification Result", font=("Helvetica", 14, "bold"), background="white", foreground="green")
        self.verification_label.place(relx=0.5, rely=0.1, anchor='center')

        self.result_label = ttk.Label(self.root, text="", font=("Helvetica", 14, "bold"), background="white", foreground="black")
        self.result_label.place(relx=0.5, rely=0.3, anchor='center')

        self.image_label = tk.Label(self.root, background="white")
        self.image_label.place(relx=0.5, rely=0.47, anchor='center')

        self.back_button = ttk.Button(self.root, text="Back", command=self.show_main_window)
        self.back_button.place(relx=0.5, rely=0.65, anchor='center', width=150, height=35)

        if predicted_label == user_name:  # Modify this condition based on your label encoding
            # messagebox.showinfo("Verification Result", f"Person Verified: {predicted_person}")
            image_path = 'tick2.png'
            audio_path = 'myoauthintro.mp3'
            self.result_label.config(text=f"Authentication success. Identified Person: {predicted_label}")
        else:
            # messagebox.showinfo("Verification Result", "Authentication Failed")
            image_path = 'wrong.png'
            audio_path = 'wrong.mp3'
            self.result_label.config(text="Authentication failed")

        original_image = Image.open(image_path)
        resized_image = original_image.resize((100, 100), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.image)

        # Play audio
        pygame.init()
        pygame.mixer.music.load(audio_path)  # Replace with your audio file path
        pygame.mixer.music.play(0, 0, 1)

        
if __name__ == "__main__":
    serial_port = "COM8"  #Serial port for Arduino
    root = tk.Tk()
    app = EMGRecorderApp(root, serial_port)

    root.mainloop()
