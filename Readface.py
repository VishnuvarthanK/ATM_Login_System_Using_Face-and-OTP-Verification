import cv2
import csv
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def take_images():
    name = name_var.get()
    Id = id_var.get()
    
    if name.isalpha() and is_number(Id):
        dict1 = {'Name': name, 'Ids': Id}
        file_exists = os.path.isfile('Profile.csv')
        
        with open('Profile.csv', 'a', newline='') as f:
            fieldnames = ['Name', 'Ids']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(dict1)
        
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier('hh.xml')
        sampleNum = 0
        
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                sampleNum += 1
                cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
            cv2.imshow('Capturing Face', img)
            
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 60:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        status_label.config(text=f"Images Saved for Name: {name} with ID: {Id}")
    else:
        messagebox.showerror("Error", "Enter a valid Name and Numeric ID")

# GUI Setup
root = tk.Tk()
root.title("Face Capture UI")
root.geometry("400x300")

tk.Label(root, text="Enter Name:").pack()
name_var = tk.StringVar()
tk.Entry(root, textvariable=name_var).pack()

tk.Label(root, text="Enter ID:").pack()
id_var = tk.StringVar()
tk.Entry(root, textvariable=id_var).pack()

tk.Button(root, text="Capture Images", command=take_images).pack()

status_label = tk.Label(root, text="", fg="green")
status_label.pack()

root.mainloop()
