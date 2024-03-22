"""import cv2
import os

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory for storing the dataset
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Initialize variables
person_id = 1  # Initialize with the ID of the first person
cv2.namedWindow("Capturing Images", cv2.WINDOW_NORMAL)

flag = True  # Flag to control the outer loop

while person_id <= 5 and flag:  # Assuming you want to capture images for 5 people
    # Create a subdirectory for the current person
    person_directory = f"dataset/person_{person_id}"
    if not os.path.exists(person_directory):
        os.makedirs(person_directory)

    count = 0  # Reset the count for each person

    while count < 200:
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop and save only the face region
            face = frame[y:y + h, x:x + w]
            count += 1
            filename = f"{person_directory}/image_{count}.jpg"
            cv2.imwrite(filename, face)

            # Display the count within the directory
            text = f"Person {person_id}, Image {count}/200"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capturing Images", frame)

            # Wait for a moment before capturing the next image (you can adjust the delay if needed)
            cv2.waitKey(100)  # 100 milliseconds

        if count >= 200:
            flag = False  # Stop capturing after 200 images for the first person
            break

    person_id += 1

cap.release()
cv2.destroyAllWindows()



import os
import cv2
import numpy as np
import random

# Set the path to the directory containing your raw face images
data_dir = "dataset"
output_dir = "preprocessed_dataset"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the image size you want for preprocessing
target_size = (128, 128)

# Define the number of augmented images per original image
num_augmentations = 5  # Adjust this number as needed

# Loop through the raw dataset and apply preprocessing steps
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(root, file)

            # Load the original image
            img = cv2.imread(img_path)

            # Resize the image to the target size
            img = cv2.resize(img, target_size)

            # Convert the image to grayscale (if needed)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0

            # Perform data augmentation
            augmented_images = []
            for _ in range(num_augmentations):
                augmented_img = img.copy()

                # Apply random augmentations
                if random.choice([True, False]):
                    augmented_img = cv2.flip(augmented_img, 1)  # Horizontal flip

                if random.choice([True, False]):
                    rotation_angle = random.uniform(-10, 10)
                    rotation_matrix = cv2.getRotationMatrix2D((target_size[0] / 2, target_size[1] / 2), rotation_angle, 1.0)
                    augmented_img = cv2.warpAffine(augmented_img, rotation_matrix, target_size, flags=cv2.INTER_LINEAR)

                # You can add more augmentations as needed

                augmented_images.append(augmented_img)

            # Save the preprocessed images (including augmented versions)
            for i, augmented_img in enumerate(augmented_images):
                output_path = os.path.join(output_dir, f"{file.replace('.jpg', f'_aug{i}.jpg')}")
                cv2.imwrite(output_path, augmented_img)

print("Data preprocessing and data augmentation complete.")


from sklearn.model_selection import train_test_split
import os
import shutil

# Set the path to your master dataset directory
dataset_dir = "dataset"  # Change this to the directory where you combined the images for each person

# Create directories for the training, validation, and testing sets
train_dir = "train"
validation_dir = "validation"
test_dir = "test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Specify the split ratios (e.g., 60% training, 20% validation, 20% testing)
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

# Get the list of image directories (one for each person)
person_subdirectories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

# Loop through the directories and split each person's images
for person_dir in person_subdirectories:
    # Get the list of image files for the current person
    image_files = [f for f in os.listdir(os.path.join(dataset_dir, person_dir)) if f.endswith(".jpg")]

    # Split the dataset for the current person
    train_files, temp_files = train_test_split(image_files, test_size=1 - train_ratio, random_state=42)
    validation_files, test_files = train_test_split(temp_files, test_size=test_ratio / (test_ratio + validation_ratio), random_state=42)

    # Create subdirectories for the current person in the train, validation, and test sets
    train_person_dir = os.path.join(train_dir, person_dir)
    validation_person_dir = os.path.join(validation_dir, person_dir)
    test_person_dir = os.path.join(test_dir, person_dir)
    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(validation_person_dir, exist_ok=True)
    os.makedirs(test_person_dir, exist_ok=True)

    # Move the files to their respective directories
    for file in train_files:
        src = os.path.join(dataset_dir, person_dir, file)
        dst = os.path.join(train_person_dir, file)
        shutil.move(src, dst)

    for file in validation_files:
        src = os.path.join(dataset_dir, person_dir, file)
        dst = os.path.join(validation_person_dir, file)
        shutil.move(src, dst)

    for file in test_files:
        src = os.path.join(dataset_dir, person_dir, file)
        dst = os.path.join(test_person_dir, file)
        shutil.move(src, dst)

print("Data splitting complete.")

import cv2
import os
import numpy as np

# Define the directory paths for your preprocessed data
train_dir = "train"
validation_dir = "validation"
test_dir = "test"

# Define the target size for resizing the images
target_size = (128, 128)


# Function to load and preprocess images
def load_and_preprocess_images(directory):
    images = []
    labels = []

    for label, person_dir in enumerate(os.listdir(directory)):
        person_path = os.path.join(directory, person_dir)

        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)

            # Load and preprocess the image
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)
            image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


# Load and preprocess training data
train_images, train_labels = load_and_preprocess_images(train_dir)

# Load and preprocess validation data
validation_images, validation_labels = load_and_preprocess_images(validation_dir)

# Load and preprocess testing data
test_images, test_labels = load_and_preprocess_images(test_dir)

# Print the shape of the datasets
print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Validation images shape:", validation_images.shape)
print("Validation labels shape:", validation_labels.shape)
print("Testing images shape:", test_images.shape)
print("Testing labels shape:", test_labels.shape)"""

import cv2
import face_recognition
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to extract face embeddings from an image
def extract_face_embeddings(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_embeddings = face_recognition.face_encodings(image, face_locations)
    return face_embeddings

# Load the embeddings for all persons
embeddings_folder = "embeddings"
persons = os.listdir(embeddings_folder)

# Initialize an empty dictionary to store embeddings
all_embeddings = {}

# Load embeddings for all persons
for person in persons:
    person_folder = os.path.join(embeddings_folder, person)
    embeddings_file_path = os.path.join(person_folder, f"{person}_embeddings.txt")

    if os.path.exists(embeddings_file_path):
        embeddings = np.loadtxt(embeddings_file_path)
        all_embeddings[person] = embeddings
        print(f"Embeddings for {person} loaded successfully.")
    else:
        print(f"Embeddings file not found for {person} in {person_folder}. Please check the folder and file names.")

# Directory containing folders for each person
dataset_path = "preprocessed"

# Create a directory for training, testing, and validation
if not os.path.exists("split_data"):
    os.makedirs("split_data")

# Create folders for training, testing, and validation
training_folder = os.path.join("split_data", "training")
testing_folder = os.path.join("split_data", "testing")
validation_folder = os.path.join("split_data", "validation")

os.makedirs(training_folder, exist_ok=True)
os.makedirs(testing_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

# List of persons in the dataset
persons = os.listdir(dataset_path)

# Prepare data for training
X_train, X_val, y_train, y_val = [], [], [], []
label_encoder = LabelEncoder()

for person, embeddings in all_embeddings.items():
    X_train.extend(embeddings)
    y_train.extend([person] * len(embeddings))

# Convert labels to numerical values
y_train = label_encoder.fit_transform(y_train)

# Train-test split for evaluation during training
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential()
model.add(Dense(128, input_dim=len(X_train[0]), activation='relu'))
model.add(Dense(len(persons), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
history = model.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_eval), np.array(y_eval)), epochs=10, batch_size=32)

# Access the training history
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_accuracy) + 1)

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Initialize the camera
#cap = cv2.VideoCapture(0)
# ... (rest of the face recognition code)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the attendance log from attendance.json
attendance_file_path = "attendance.json"
if not os.path.exists(attendance_file_path):
    with open(attendance_file_path, "w") as f:
        json.dump({}, f)

# Initialize attendance_log as an empty dictionary
attendance_log = {}

# Create variables for accuracy calculation
total_faces = 0
correctly_recognized_faces = 0

recognized_faces = []  # List to store recognized faces with timestamps

# Lists to store metrics for each recognized face
precision_list = []
recall_list = []
f1_list = []

# ... (previous code)

# Create variables for accuracy calculation
total_faces = 0
correctly_recognized_faces = 0

recognized_faces = []  # List to store recognized faces with timestamps

# Lists to store metrics for each recognized face
precision_list = []
recall_list = []
f1_list = []

while True:
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        total_faces += 1
        recognized = False

        # Check if the face matches any person
        for person, embeddings in all_embeddings.items():
            face_distances = face_recognition.face_distance(embeddings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.4:  # Adjust the threshold if needed
                recognized = True
                name = person

                # Print debugging information
                print(f"Recognized {name} with distance {face_distances[best_match_index]}")

                # Check if the recognized person is in the ground truth dataset
                image_path = os.path.join("dataset", name, f"{name}_image_{total_faces}.jpg")
                if os.path.exists(image_path):
                    correctly_recognized_faces += 1

                    # Visualize recognized face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Break after recognizing a face to avoid assigning multiple names to the same face
                break

        # Display "Unknown" if the face is not recognized
        if not recognized:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Calculate and print overall precision, recall, and f1-score after processing all faces
precision = correctly_recognized_faces / total_faces
recall = correctly_recognized_faces / total_faces
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Overall Precision: {precision:.2f}")
print(f"Overall Recall: {recall:.2f}")
print(f"Overall F1-score: {f1:.2f}")

# Convert the recognized faces list to a DataFrame
df_recognized_faces = pd.DataFrame(recognized_faces)

# Save the DataFrame to an Excel file
excel_file_path = "recognized_faces.xlsx"
df_recognized_faces.to_excel(excel_file_path, index=False)
print(f"Excel file saved at: {excel_file_path}")

# Release the camera
cap.release()
cv2.destroyAllWindows()



"""import matplotlib.pyplot as plt
import seaborn as sns

# Load the attendance log from attendance.json
with open(attendance_file_path, "r") as f:
    attendance_log = json.load(f)

# Convert the attendance log to a DataFrame
df_attendance = pd.DataFrame(list(attendance_log.items()), columns=["Name", "Timestamp"])

# Extract date and time components from the Timestamp column
df_attendance["Date"] = pd.to_datetime(df_attendance["Timestamp"]).dt.date
df_attendance["Time"] = pd.to_datetime(df_attendance["Timestamp"]).dt.time

# Count the number of occurrences for each date
date_counts = df_attendance["Date"].value_counts().sort_index()

# Plotting the attendance over time
plt.figure(figsize=(10, 6))
sns.barplot(x=date_counts.index, y=date_counts.values, color='blue')
plt.title('Attendance Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Attendees')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as an image
attendance_plot_path = "attendance_plot.png"
plt.savefig(attendance_plot_path)

# Display the plot
plt.show()

print(f"Attendance plot saved at: {attendance_plot_path}")


# Pie chart for the distribution of recognized faces
plt.figure(figsize=(8, 8))
labels = ['Correctly Recognized', 'Unknown']
sizes = [correctly_recognized_faces, total_faces - correctly_recognized_faces]
colors = ['green', 'red']
explode = (0.1, 0)  # explode the 1st slice (Correctly Recognized)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribution of Recognized Faces')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

# Save the pie chart as an image
pie_chart_path = "pie_chart.png"
plt.savefig(pie_chart_path)

# Display the pie chart
plt.show()

print(f"Pie chart saved at: {pie_chart_path}")

# Histogram for accuracy
plt.figure(figsize=(8, 6))
plt.hist([accuracy], bins=10, color='blue', edgecolor='black')
plt.title('Accuracy Histogram')
plt.xlabel('Accuracy (%)')
plt.ylabel('Frequency')

# Save the histogram as an image
accuracy_hist_path = "accuracy_histogram.png"
plt.savefig(accuracy_hist_path)

# Display the histogram
plt.show()

print(f"Accuracy histogram saved at: {accuracy_hist_path}")

import matplotlib.pyplot as plt
import seaborn as sns

# ... (previous code)

# Bar chart for the distribution of recognized faces
plt.figure(figsize=(8, 6))
sns.barplot(x=['Correctly Recognized', 'Unknown'], y=[correctly_recognized_faces, total_faces - correctly_recognized_faces], palette=['green', 'red'])
plt.title('Distribution of Recognized Faces')
plt.ylabel('Number of Faces')
plt.tight_layout()

# Save the bar chart as an image
bar_chart_path = "bar_chart.png"
plt.savefig(bar_chart_path)

# Display the bar chart
plt.show()

print(f"Bar chart saved at: {bar_chart_path}")

# Scatter plot for accuracy over time
plt.figure(figsize=(12, 6))
plt.scatter(df_attendance["Timestamp"], df_attendance.groupby("Date").size(), color='blue', label='Correctly Recognized Faces', s=50)
plt.title('Accuracy Over Time (Scatter Plot)')
plt.xlabel('Date')
plt.ylabel('Number of Correctly Recognized Faces')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the scatter plot as an image
scatter_plot_path = "scatter_plot.png"
plt.savefig(scatter_plot_path)

# Display the scatter plot
plt.show()

print(f"Scatter plot saved at: {scatter_plot_path}")

# Release the camera
cap.release()
cv2.destroyAllWindows()

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ... (previous code)

# Create a bar chart for the distribution of images in training, testing, and validation sets
plt.figure(figsize=(10, 6))

# Count the number of images in each set
train_counts = [len(os.listdir(os.path.join("data", "training", person))) for person in persons]
test_counts = [len(os.listdir(os.path.join("data", "testing", person))) for person in persons]
val_counts = [len(os.listdir(os.path.join("data", "validation", person))) for person in persons]

# Calculate the total number of images
total_train = sum(train_counts)
total_test = sum(test_counts)
total_val = sum(val_counts)

# Plot the bar chart
sns.barplot(x=["Training", "Testing", "Validation"], y=[total_train, total_test, total_val], palette="viridis")
plt.title('Distribution of Images in Training, Testing, and Validation Sets')
plt.ylabel('Number of Images')
plt.tight_layout()

# Save the bar chart as an image
set_distribution_chart_path = "set_distribution_chart.png"
plt.savefig(set_distribution_chart_path)

# Display the bar chart
plt.show()

print(f"Set distribution chart saved at: {set_distribution_chart_path}")
