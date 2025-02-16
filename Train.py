import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def train_images():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, Ids = getImagesAndLabels("TrainingImage")
        recognizer.train(faces, np.array(Ids))
        recognizer.save("TrainData/Trainner.yml")
        status_label.config(text="Training Completed and Data Stored!", fg="green")
    except Exception as e:
        messagebox.showerror("Error", f"Training Failed: {str(e)}")

# GUI Setup
root = tk.Tk()
root.title("Face Training UI")
root.geometry("400x200")

tk.Label(root, text="Face Training System", font=("Arial", 14)).pack(pady=10)
train_button = tk.Button(root, text="Train Images", command=train_images)
train_button.pack()

status_label = tk.Label(root, text="", fg="blue")
status_label.pack()

root.mainloop()