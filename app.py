from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd
from flask import send_file
from flask import render_template, redirect, url_for, request

app = Flask(__name__)

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

# Initialize the camera
video_capture = cv2.VideoCapture(0)

# Load the attendance log from attendance.json
attendance_file_path = "attendance.json"
if not os.path.exists(attendance_file_path):
    with open(attendance_file_path, "w") as f:
        json.dump({}, f)

# Initialize attendance_log as an empty dictionary
attendance_log = {}

# Create global variables for accuracy calculation
total_faces = 0
correctly_recognized_faces = 0

recognized_names = set()
recognized_faces = []  # List to store recognized faces with timestamps

video_processing = True

def gen_frames():
    global total_faces, correctly_recognized_faces, video_processing, video_capture  # Declare these as global

    while True:
        if video_processing:
            success, frame = video_capture.read()
            if not success:
                break

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            #print(f"Number of faces detected: {len(face_locations)}")

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                total_faces += 1
                recognized = False

                # Check if the face matches any person
                for person, embeddings in all_embeddings.items():
                    face_distances = face_recognition.face_distance(embeddings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    #print(f"Face distance for {person}: {face_distances[best_match_index]}")

                    if face_distances[best_match_index] < 0.45:  # Adjust the threshold if needed
                        recognized = True
                        name = person

                        print(f"Recognized {name}")

                        # Check if the recognized person is in the ground truth dataset
                        image_path = os.path.join("dataset", name, f"{name}_image_{total_faces}.jpg")
                        if os.path.exists(image_path):
                            correctly_recognized_faces += 1

                            # Mark attendance
                            if name not in attendance_log:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                attendance_log[name] = timestamp

                                # Store recognized face information
                                recognized_faces.append({"Name": name, "Timestamp": timestamp})

                                # Write the updated attendance log to the JSON file
                                try:
                                    with open(attendance_file_path, "w") as f:
                                        json.dump(attendance_log, f)
                                except Exception as e:
                                    print(f"Error writing to JSON file: {e}")

                            # Draw rectangle and name on the frame
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Break after recognizing a face to avoid assigning multiple names to the same face
                        break

                # Display "Unknown" if the face is not recognized
                if not recognized:
                    print("Face not recognized")
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check login credentials (replace this with your authentication logic)
        username = request.form.get('username')
        password = request.form.get('password')

        # Example: Check if username and password match
        if username == 'Nikhila' and password == 'sru123':
            # Redirect to the main page on successful login
            return redirect(url_for('index'))

        # Redirect back to the login page on failed login
        return redirect(url_for('login'))

    # Render the login page for GET requests
    return render_template('login.html')

@app.route('/')
def index():
    # Check if the user is authenticated (replace this with your authentication logic)
    authenticated = True  # Replace this with your actual authentication check

    if not authenticated:
        # Redirect to the login page if not authenticated
        return redirect(url_for('login'))

    # Render the main page for authenticated users
    return render_template('index.html')

@app.route('/start_video')
def start_video():
    global video_processing
    video_processing = True
    return 'Video started'

@app.route('/stop_video')
def stop_video():
    global video_processing
    video_processing = False
    return 'Video stopped'

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_excel')
def download_excel():
    global recognized_faces

    # Convert the recognized faces list to a DataFrame
    df_recognized_faces = pd.DataFrame(recognized_faces)

    # Save the DataFrame to an Excel file
    excel_file_path = "recognized_faces.xlsx"
    df_recognized_faces.to_excel(excel_file_path, index=False)

    # Send the file as an attachment with the appropriate MIME type
    return send_file(excel_file_path, as_attachment=True, download_name='recognized_faces.xlsx')

if __name__ == '__main__':
    app.run(debug=True)
