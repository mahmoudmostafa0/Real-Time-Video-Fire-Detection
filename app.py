from ast import Try
from cv2 import COLOR_BGR2RGB
from pygame import mixer
from requests import request
from sqlalchemy import true
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
from flask import Flask, redirect, url_for, jsonify,Response,request,make_response,send_from_directory,render_template
import cv2
import numpy as np
import time
import threading
from detection_case import DetectionCase
from datetime import datetime
app = Flask(__name__)
detection_path = app.static_folder + "/detection/"
detected_fires = {}
cur_frame = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(
        './best (8).pt', map_location=device)


def detection_thread():
    global cur_frame
    cap = cv2.VideoCapture("./cctv.mp4")  # 0 is for webcam ./cctv.mp4
    while True:
        highest_accuracy_frame = None
        highest_accuracy_pct = 0.0
        positive_detections = 0.0
        for i in range(30):
            #time.sleep(0.1)
            ret, frame = cap. read()
            cur_frame = frame
            highest_conf,frame = detect_frame(frame)
            if highest_conf != 0.0:
                positive_detections+=1
            if highest_conf > highest_accuracy_pct:
                highest_accuracy_pct = highest_conf
                highest_accuracy_frame = frame
        accuracy_for_30_frames = positive_detections/30
        print(accuracy_for_30_frames)
        if accuracy_for_30_frames > 0.5:
            mixer.init() 
            sound=mixer.Sound("alarmhigh.wav")
            sound.play()
            file_name = str(datetime.now().strftime("%Y-%m-%d_%I-%M-%S %p"))
            print(file_name)
            case = DetectionCase(file_name+".jpg",float(highest_accuracy_pct),)
            detected_fires[file_name+".jpg"]=(case)
            cv2.imwrite(detection_path + "/"+file_name +
                        ".jpg", highest_accuracy_frame)
            cv2.imshow('', highest_accuracy_frame)
            cv2.waitKey(1)

def detect_frame(frame):
    global cur_frame
    frame = cv2.resize(frame, (640, 640))
    image = cv2.cvtColor(frame, COLOR_BGR2RGB)
    #image = cv2.resize(image,(640,640))
    image = np.array(image)
    image = image / 255.0
    image = torch.from_numpy(np.array(image)).permute(
        2, 0, 1).unsqueeze(0).float()
    image = image.to(device)
    # model.eval()
    inf_out, tt = model(image)
    output = non_max_suppression(
        inf_out, conf_thres=0.3, iou_thres=0.6)
    for det in output:
        if len(det):
            highest_conf =0.0
            for pred in det:
                box = pred[:4]
                w = (box[2] - box[0])
                h = (box[3] - box[1])
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int((box[0] + w).item())
                y2 = int((box[1] + h).item())
                conf = float(pred[4].item())
                if conf > highest_conf:
                    highest_conf = conf
                cls = int(pred[5])
                cv2.rectangle(frame, (x1, y1),
                                (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, str(conf), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (125, 246, 55))

            cur_frame = frame
            return highest_conf,frame
        return 0.0,None

@app.route('/display/<filename>')
def display_image(filename):
    pathh = app.root_path + "/detection/" + filename
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename="detection/"+filename), code=301)


    resp = make_response(send_from_directory(app.static_folder, "firebase-messaging-sw.js"))
    resp.headers["content-type"]= "text/javascript;"
    return resp

def live_frames():
    global cur_frame
    while True:
        ret, jpeg = cv2.imencode('.jpg', cur_frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/live')
def video_feed():
    return Response(live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route("/viewnotchecked")
def viewnotchecked():
    return jsonify([x.ToJson() for x in detected_fires.values() if x.GotFeedback == False])
@app.route("/viewhistory")
def viewhistory():
    return jsonify([x.ToJson() for x in detected_fires.values() if x.GotFeedback == True])
@app.route("/")
def main():
    return redirect(url_for('static', filename="home.html"), code=301)
@app.route("/sendfeedback")
def SendFeedback():
    imgname = request.args["imgname"]
    if imgname not in detected_fires:
        return "Cannot find image"
    case = detected_fires[imgname]
    feedback = eval(request.args["Feedback"])
    if case.GotFeedback:
        return "Case Was Already Updated"
    case.FeedbackResult =feedback
    case.GotFeedback = True
    return redirect("/alarms")

@app.route("/loadimgs")
def loadimgs():
    imageshtml = ""
    for x in detected_fires.values():
        if x.GotFeedback == true:
            continue
        imageshtml += ("<div class=\"imgContainer\">"  +
            "<img src=\"\static\detection\\"+x.Path+"\"/>" +
            "<div class=\"imgButton\">" +
            "<form action=\"/sendfeedback\" method=\"get\">" +
            "<input type=\"hidden\" id=\"imgname\" name=\"imgname\" value=\""+x.Path+"\">"+
            "<input type=\"submit\" name=\"Feedback\" value=\"True\">" +
            "<input type=\"submit\" name=\"Feedback\" value=\"False\">" +
            "</form></div></div>")
    return imageshtml

@app.route("/alarms")
def alarms():
    return redirect(url_for('static', filename="alarms.html"), code=301)


@app.route("/testurl")
def test_func():
    return jsonify([
        {"imgname": "2022-06-26_10-40-54 PM.jpg", "pct": "0.8"},
        {"imgname": "2022-06-26_10-48-57 PM.jpg", "pct": "0.5"},
        {"imgname": "2022-06-26_10-48-57 PM.jpg", "pct": "0.5"},
        {"imgname": "2022-06-26_10-48-57 PM.jpg", "pct": "0.5"},
        {"imgname": "2022-06-26_10-48-57 PM.jpg", "pct": "0.5"},
        {"imgname": "2022-06-26_10-48-57 PM.jpg", "pct": "0.5"},
        {"imgname": "2022-06-26_10-48-57 PM.jpg", "pct": "0.5"},
        {"imgname": "2022-06-26_10-48-57 PM.jpg", "pct": "0.5"}

    ])


if __name__ == "app":
    # app.run(debug=True)
    detection_thread = threading.Thread(target=detection_thread, daemon=True)
    detection_thread.start()
