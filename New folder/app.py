import os
import cv2
import face_recognition
import numpy as np
import datetime
from flask import Flask, render_template, request, Response, redirect, url_for
from werkzeug.utils import secure_filename
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Email Config
SENDER_EMAIL = "anandkarthik2002@gmail.com"
SENDER_PASSWORD = "anand23102002"  # Use Gmail App Password

def send_email_alert(recipient_email, person_name):
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = recipient_email
    message["Subject"] = f"Match Found for {person_name}"

    body = f"Hello,\n\nA face match has been found for {person_name}."
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplpl.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, message.as_string())
        server.quit()
        print(f"[INFO] Email sent to {recipient_email}")
    except Exception as e:
        print(f"[ERROR] Email failed: {e}")

def extract_frame(video_path, frame_number, save_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = min(frame_number, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()

    if success:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)
        print(f"[INFO] Extracted frame saved to {save_path}")
        return True
    else:
        print("[ERROR] Frame extraction failed.")
        return False

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/extract', methods=['POST'])
def extract():
    name = request.form['name']
    email = request.form['email']
    age = int(request.form['age'])

    video = request.files['video']
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.mp4')
    video.save(video_path)

    save_dir = os.path.join('dataset', name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}.jpg')

    frame_number = min(max(age, 0), 50)
    success = extract_frame(video_path, frame_number, save_path)

    if success:
        image_path = os.path.join('/dataset', name, f'{name}.jpg')
        return render_template('home.html', image_path=image_path, name=name, email=email)
    else:
        return "Frame extraction failed"

def gen_frames(known_encoding, name, email):
    cap = cv2.VideoCapture(0)
    match_sent = False

    matched_folder = os.path.join('matched', name)
    os.makedirs(matched_folder, exist_ok=True)

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces([known_encoding], face_encoding)
            if matches[0]:
                label = name
                color = (0, 255, 0)

                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                matched_img_path = os.path.join(matched_folder, f'{name}_{timestamp}.jpg')
                cv2.imwrite(matched_img_path, frame)

                if not match_sent:
                    send_email_alert(email, name)
                    match_sent = True
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    name = request.form['name']
    email = request.form['email']
    image_path = os.path.join('dataset', name, f'{name}.jpg')

    if not os.path.exists(image_path):
        return "User image not found."

    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if not encoding:
        return "No face found in extracted image."
    
    known_encoding = encoding[0]
    return Response(gen_frames(known_encoding, name, email), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
