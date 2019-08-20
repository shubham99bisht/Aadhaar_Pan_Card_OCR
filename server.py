import cv2
from demo_pb_new import CTPN
ctpn = CTPN()
import pytesseract
import json
# from regex_code_pan import *

# from scan_new import DocScanner
# scanner = DocScanner()
#-------------------------------------

import flask
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import ast, io, os, shutil, time
from PIL import Image
import numpy as np
app = Flask(__name__)

#-------------------------------------------------------------------------------------------
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/invoice_demo")
def invoice_demo():
    return render_template("invoice_demo.html")

@app.route("/cheque_demo_ocr/<id>")
def cheque_demo_ocr(id):
    return render_template("ocr.html", image_name=id)

#-------------------------------------------------------------------------------------------

@app.route('/show_image/<filename>')
def show_image(filename):
    filename += "_crop.png"
    return send_from_directory("uploads/",(filename))


@app.route('/crop/<id>')
def crop(id):
    return render_template("crop.html", image_name=id)

@app.route('/crop_image/<filename>')
def crop_image(filename):
    return send_from_directory("uploads/", filename+".png")


@app.route("/invoice_upload", methods=["POST"])
def invoice_upload():
    if request.method == "POST":
        f= request.files["image"]
        filename = secure_filename(f.filename)
        basedir = os.path.abspath(os.path.dirname(__file__))
        id = "i" + str(int(time.time()))
        save_name = id + ".png"
        f.save(os.path.join(basedir, "uploads", save_name))
        return render_template("crop.html", image_name=id)


@app.route('/confirm_crop/<id>', methods=["POST"])
def confirm_crop(id):
    if request.method == "POST":
        # receiving coordinates to crop
        x1 = request.args.get("x1")
        y1 = request.args.get("y1")
        x2 = request.args.get("x2")
        y2 = request.args.get("y2")
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

        imgcv = cv2.imread("uploads/"+id+".png")
        imgcv = imgcv[y1:y2,x1:x2]

        cv2.imwrite("uploads/"+id+"_crop.png", imgcv)

        all_boxes = ctpn.main2(imgcv,"uploads/"+id+"_crop.png")    # CTPN output boxes, to get original image size, use img.shape

        final_dict = Tesseract_Line(imgcv, all_boxes,id)   # passing all boxes to Tesseract_Line function defined below
        return render_template("results.html", result=final_dict, image_name=id)


def Tesseract_Line(imgcv, all_boxes, fbase):
    ret_list = []
    ctr = 0
    for box in sortBoxes(all_boxes):
        ctr += 1
        clone = imgcv.copy()
        cropped = clone[box[1]:box[3],box[0]:box[2]]
        cropped = cv2.resize(cropped, (int(cropped.shape[1]*1.5),int(cropped.shape[0]*1.5)), interpolation = cv2.INTER_AREA)
        cv2.imwrite("cutfolder/fbase"+str(ctr)+".jpg",cropped)
        label = pytesseract.image_to_string(cropped,config="--psm 7")
        ret_list.append(box+[label])
    for line in ret_list:
        print(line[4])
    # final_dict = getData(ret_list)
    return ret_list


#-------------------------------------------------------------------------------------------
# OCR for NACH document

@app.route("/ocr_upload", methods=["POST"])
def ocr_upload():
    if request.method == "POST":
        f= request.files["image"]
        doc_type = request.form["doc_type"]   # can receive two values 1. cheque 2. nach
        print(doc_type)
        filename = secure_filename(f.filename)
        basedir = os.path.abspath(os.path.dirname(__file__))
        id = "o" + str(int(time.time()))
        save_name = id + ".png"
        filepath = os.path.join(basedir, "uploads", save_name)
        f.save(filepath)
        # preprocess(filepath, doc_type)
        return render_template("ocr.html", image_name=id)

def preprocess(filepath, doc_type):
    # Do some preprocessing based on document type, dont change the name of file. file name is the id which will be used to reference it.
    pass

@app.route("/ocr/<id>")
def ocr(id):
    # Receives the four coordinates from user and returns ocr detected text, also receives the "type" value which can be used for postprocessing.
    x1 = request.args.get("x1")
    y1 = request.args.get("y1")
    x2 = request.args.get("x2")
    y2 = request.args.get("y2")
    postprocessing_type = int(request.args.get("type"))

    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
    img = cv2.imread("uploads/{}.png".format(id))
    crop = img[y1:y2,x1:x2]
    cv2.imwrite("ocr/temp.png", crop)
    data = postprocessing(crop,"ocr/temp.png", postprocessing_type)
    # text = g_ocr.detect_document_line("ocr/temp.png")
    # text = postprocessing(text, postprocessing_type)
    # data = {"text":text}
    return json.dumps(data)


def postprocessing(imgcv, cropped_img_path, postprocessing_type):
    # postprocessing_type_dictionary = {1: "name", 2: "amount in words", 3: "amount in digits", 4: "dates", 5: "number", 6: "text", 7:"pan", 8: "aadhar", 9:"year", 10: "micr"}
    # postprocessing_type will be an integer value which will be a key in the above dictionary
    text = g_ocr.detect_document_line(cropped_img_path)   # Using Google OCR by default for all types
    data = {"text":text}
    return data
