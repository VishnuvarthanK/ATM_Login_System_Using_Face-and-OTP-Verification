from flask import Flask, request, jsonify
import random
import os
import webbrowser
from twilio.rest import Client
from flask_cors import CORS
import cv2, shutil, csv
import numpy as np
import pandas as pd
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Twilio Credentials (Replace with actual credentials)
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE_NUMBER = ""
USER_PHONE_NUMBER = ""  # Replace with actual user phone

otp_store = {}  # Temporary storage for OTPs

# Generate and send OTP
def send_otp(phone):
    otp = str(random.randint(100000, 999999))
    otp_store[phone] = otp  # Store OTP for verification
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=f"Your ATM login OTP is: {otp}",
        from_=TWILIO_PHONE_NUMBER,
        to=phone
    )
    return message.sid

@app.route('/send_otp', methods=['POST'])
def send_otp_route():
    data = request.json
    card_number = data.get("card_number")
    if not card_number:
        return jsonify({"message": "Card number is required!"}), 400
    
    sid = send_otp(USER_PHONE_NUMBER)
    return jsonify({"message": "OTP sent successfully!", "sid": sid})

@app.route('/verify_otp', methods=['POST'])
def verify_otp_route():
    data = request.json
    user_otp = data.get("otp")
    if otp_store.get(USER_PHONE_NUMBER) == user_otp:
        return jsonify({"message": "Login Successful!", "redirect": "dashboard.html"})
    return jsonify({"message": "Invalid OTP!"}), 400

# Face Recognition and Data Storage
def clean_profile_csv():
    if os.path.exists("Profile.csv"):
        df = pd.read_csv("Profile.csv", encoding="utf-8")
        df.sort_values("Ids", inplace=True)
        df.drop_duplicates(subset="Ids", keep="first", inplace=True)
        df.to_csv("Profile.csv", index=False)
    else:
        print("Error: Profile.csv not found!")
        exit()

@app.route('/run-detectface', methods=['GET'])
def detect_face():
    clean_profile_csv()
    name_dict = {"Unknown": "Unknown"}

    try:
        with open("Profile.csv", "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for rows in reader:
                name_dict[rows["Ids"]] = rows["Name"]
    except Exception as e:
        return jsonify({"error": f"Error reading Profile.csv: {e}"})

    if not os.path.exists("TrainData/Trainner.yml"):
        return jsonify({"error": "TrainData/Trainner.yml not found!"})

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainData/Trainner.yml")

    harcascadePath = "hh.xml"
    if not os.path.exists(harcascadePath):
        return jsonify({"error": f"{harcascadePath} not found!"})

    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return jsonify({"error": "Cannot access webcam!"})

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cam.read()
        if not ret:
            return jsonify({"error": "Cannot read from webcam!"})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        face_id = "Not detected"
        color = (0, 0, 255)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 60:
                name = name_dict.get(str(Id), "Unknown")
                face_id = name
                color = (0, 255, 0)
            else:
                face_id = "Unknown"

            cv2.putText(frame, str(face_id), (x, y - 10), font, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if face_id != "Unknown" and face_id != "Not detected":
            cam.release()
            cv2.destroyAllWindows()
            webbrowser.open("dashboard.html")
            return jsonify({"success": f"WELCOME {face_id}! Login Successful!"})

    cam.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "No face detected"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
