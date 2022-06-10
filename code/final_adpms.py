‹import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2
from tkinter import *
import csv
from skimage import io
from datetime import datetime
from datetime import date
import time
import json
import requests
import os
#from datetime import datetime
import matplotlib.pyplot as plt
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pywhatkit as kit

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'object_detection\images\labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

import pathlib
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
#TEST_IMAGE_PATHS

### change the location accordingly
detection_model = tf.saved_model.load('object_detection/inference_graph/saved_model')

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
#   print(output_dict)
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
#   print(output_dict['detection_classes'])
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.8,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def show_inference(model, image_np):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
#   image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)

#   print(category_index)
  # Visualization of the results of a detection.
  final_img =vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=8)
  return(final_img)
#   display(Image.fromarray(image_np))

today = date.today()
d1 = today.strftime("%d-%b-%Y")

filename=[]

access = False

# 1AFFAB or 61FFA3

bgc = '#1AFFAB'
btc = '#EFC88A'
ent = '#FFFFFF' 
#LOGIN
rooting =Tk()
rooting.title('Login to Climate Chnage Detector.')
fg = '#FFFFFF'

rooting.geometry('850x650')
rooting.configure(background=fg)
bg = PhotoImage(file='ool.png')
label=Label(rooting, image = bg)
label.place(x=0, y=1)

userlogin='0'

lab = Label(text ='Welcome to CCD (M&A) ',bg = fg, font=(None, 25, 'bold'))
lab.place(x = 363, y = 50)

lab1 = Label(text = 'Employee Name',bg = fg, fg = '#646464', font=(None, 17))
lab1.place(x= 400, y= 160)
en1 = Entry(width=50, bg = ent, fg='#778899', font=(None, 15) )
en1.place(x=400, y=200, width = 320)

lab2 = Label(text = 'Employee ID',bg = fg, fg = '#646464', font=(None, 17))
lab2.place(x= 400, y= 260)
en2 = Entry(width=50, bg = fg, fg='#778899', font=(None, 15))
en2.place(x=400, y=300, width = 320)

lab3 = Label(text = 'Location',bg = fg, fg ='#646464', font=(None, 17))
lab3.place(x= 400, y= 360)
en3 = Entry(width=50, bg = fg, fg='#778899', font=(None, 15))
en3.place(x=400, y=400, width = 320)

lab4 = Label(text='Made by © Arnav Suman | All Rights Reserved',bg = fg, font=(None, 12, 'bold'))
lab4.place(x=380, y=620)

t2 = time.localtime()
time_current = time.strftime("%H.%M.%S", t2)

def nameerror():
    tk.messagebox.showerror("Name Field Empty ", "Please enter your Name.")

def iderror():
    tk.messagebox.showerror("Employee ID Field Empty ", "Please enter your Employee ID.")

def locationerror():
    tk.messagebox.showerror("Location Field Empty ", "Please enter Location you are right now.")

def login():
    emp_name = en1.get()
    emp_name = emp_name.lower()
    global employee_fullname
    employee_fullname = str(emp_name)
    user_name = ''
    emp_id = en2.get()
    emp_id = emp_id.lower()
    user_id = ''
    global location
    location = en3.get()
    location = location.lower()
    users={
        "arnavsuman":"test101",
        "sahilveshin":"test102",
        "sambhavpeshin":"test103"
    }
    disturbance = ['',' ',',','.','/',"'",'[',']','{','}','(',')','+','-','*','!','@','#','$','&']
    for ch in emp_name:
        if ch in disturbance:
            pass
        else:
            user_name = user_name + ch
    for ca in emp_id:
        if ca in disturbance:
            pass
        else:
            user_id = user_id + ca
    if user_name not in users.keys():
        print('Please Enter Correct Employee Name! ')
        nameerror()
    elif user_id not in users.values():
        print('Please Enter Correct Employee ID! ')
        iderror()
    elif location == '':
        print('Please Enter Location First. ')
        locationerror()
    else:
        global access
        access = True
        fcsv = open('ADPMS.csv', 'a', newline='')
        writer = csv.writer(fcsv)
        tup1=("NAME","ID","LOCATION","Date","Log in Time")
        tup2=(emp_name, emp_id, location, d1, time_current)
        writer.writerow(tup1)
        writer.writerow(tup2)
        writer.writerow("")
        en1.delete(0, END)
        en2.delete(0, END)
        en3.delete(0, END)
        fcsv.close()
        global userlogin
        userlogin='1'
        rooting.destroy()

but = Button(text = 'Login Now', fg = 'black', bg = '#78ffd6',font=(None, 25), command = login).place(x = 450, y = 490)

rooting.mainloop()

# LOGIN ENDS
#MAIN SCREEN
if access == True:
    root=Tk()
    root.geometry('1500x800')
    root.title("Welcome to Climate Chnage Detector! "+ employee_fullname)
    root.configure(background = bgc)
    def graph():
        roots = Tk()
        f = open('ADPMS.csv', 'r')
        read = csv.reader(f)
        read = list(read)
        f.close()
        mumbai = 0
        delhi = 0
        kolkata = 0
        bangalore = 0
        noida = 0 
        for i in read:
            if ('noida') in i: 
                noida=noida+1
                
            if ('delhi') in i:
                delhi=delhi+1
                
            if ('mumbai') in i:
                mumbai=mumbai+1
                
            if ('bangalore') in i:
                bangalore=bangalore+1
                
            if ('kolkata') in i:
                kolkata=kolkata+1

        x1 = ['NOIDA', 'DELHI', 'BANGALORE', 'KOLKATA', 'MUMBAI']
        y1 = [noida, delhi, bangalore, kolkata, mumbai]
        
        data_pol = {'NAME of CITY': x1,
            'No. of Incidents': y1
            }  
        df_pol = DataFrame(data_pol,columns=['NAME of CITY', 'No. of Incidents'])
        figure = plt.Figure(figsize=(9, 7), dpi=100)
        ax = figure.add_subplot(111)
        ax.scatter(df_pol['NAME of CITY'], df_pol['No. of Incidents'], color = 'r')
        scatter = FigureCanvasTkAgg(figure, roots) 
        scatter.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        ax.legend(['No. of Pollution Violations']) 
        ax.set_xlabel('<-----------NAME of CITY----------->')
        ax.set_ylabel('<-------Pollution Violations------->')
        ax.set_title('Cities with Highest Pollution Violation')
        roots.mainloop()

    def ml():
        if access == True:
            lb381 = Label(text='Time: ' + time_current + ' Hrs', bg = bgc, fg=bgc, font=('Bahnschrift Light Condensed', 38, 'bold'))
            lb381.place(x=1050, y=260)
            t2a = time.localtime()
            time_currenta = time.strftime("%H.%M.%S", t2a)
            lb371 = Label(text='Time: ' + time_currenta + ' Hrs', bg = bgc, font=('Bahnschrift Light Condensed', 38, 'bold'))
            lb371.place(x=1050, y=260) 
            
            count=0
            counter=0
            asource = clicked.get()
            if asource == 'DRONE / WEBCAM':
                source = 0
                source = int(source)
            elif asource == 'Sample 1 Clean':
                source = '1.mp4'
            elif asource == 'Sample 1 Pollution':
                source = '1_pol.mp4'
            elif asource == 'Sample 2 Clean':
                source = '2.mp4'
            elif asource == 'Sample 2 Pollution':
                source = '2_pol.mp4'
            elif asource == 'Sample 3 Pollution':
                source = '3_pol.mp4'
            else:
                source = '2_pol.mp4'
            cap = cv2.VideoCapture(source)
            #cap = cv2.VideoCapture('2_pol.mp4') 
            #Switch between filenames or 0,1 for webcam feeds.'1.mp4'or'1_pol.mp4' or '2.mp4' or '2_pol.mp4' or '3_pol.mp4'
            cap.set(3, 1000)
            cap.set(4, 1000)

            t1 = time.localtime()
            time_now = time.strftime("%H.%M.%S", t1)
            video_name = 'Image/videos/Backup_Video_file_'+d1+'_'+time_now+'_'+str(counter)+'.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            rec = cv2.VideoWriter(video_name, fourcc, 5, (640, 480))
            while True:
                ret, img = cap.read()
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                final_img = show_inference(detection_model,img)
                final_img = cv2.cvtColor(final_img,cv2.COLOR_RGB2BGR)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(final_img, str(datetime.now()), (10,30), font, 1, (255,255,255),2,cv2.LINE_AA)
                cv2.putText(final_img, location, (10,80), font, 1, (255,255,255),2,cv2.LINE_AA)
                rec.write(final_img)
                cv2.imshow('Press Esc/Space to Pause Video Feed/Capture Image',final_img)

                k=cv2.waitKey(1)
                if k%256==27: #(esc key to quit)
                    rec.release()
                    counter+=1
                    video_name = video_name[6:]
                    filename.append(video_name)

                    break
                elif k%256==32: # space to capture
                    print('image saved')
                    file= "Image/img" +'-'+d1+'-'+time_now+'-'+str(count)+'.jpg'
                    cv2.imwrite(file, final_img)
                    count+=1
                    img_name = file[6:]
                    filename.append(img_name)
            cap.release()
            but = Button(text='Graph Ready', font=(None, 22), bg='#EFC88A', fg='black', command = graph)
            but.place(x=1045, y=410)
        else:
            tk.messagebox.showerror("Invalid Login Credentials ", "Your Login was incomplete. Log in Again to use Launch Webcam ")

    lb2 = Label(text='Date: '+d1, bg = bgc, font=('Bahnschrift Light Condensed', 38, 'bold'))
    lb2.place(x=1050, y=175)
    t2 = time.localtime()
    time_current = time.strftime("%H.%M.%S", t2)
    lb3 = Label(text='Time: ' + time_current + ' Hrs', bg = bgc, font=('Bahnschrift Light Condensed', 38, 'bold'))
    lb3.place(x=1050, y=260) 

    lb = Label(text=' Welcome to CCD(M&A), '+ employee_fullname, bg = bgc, font=('Bahnschrift Light Condensed', 50, 'bold'))
    lb.place(x=10, y=10)

    time_current_int = int(time_current[0:2])
    
    if time_current_int < 12:
        lx = Label(text='Good Morning!', bg = bgc, font=('Bahnschrift Light Condensed', 40, 'bold'))
        lx.place(x=50, y=150)
        
    elif time_current_int in range(12, 19):
        llx = Label(text='Good Afternoon!', bg = bgc, font=('Bahnschrift Light Condensed', 40, 'bold'))
        llx.place(x=50, y=150)
    else:
        lhx = Label(text='Good Evening!', bg = bgc, font=('Bahnschrift Light Condensed', 40, 'bold'))
        lhx.place(x=50, y=150)

    options = [
        'DRONE / WEBCAM',
        'Sample 1 Clean',
        'Sample 1 Pollution',
        'Sample 2 Clean',
        'Sample 2 Pollution',
        'Sample 3 Pollution'
    ]
    clicked = StringVar()
    clicked.set(options[4])

    lb555 = Label(text='Select Video Feed: ', fg = '#4B5230', bg = bgc, font = ('Bahnschrift Light Condensed', 35, 'bold'))
    lb555.place(x = 100, y = 400)#'black'
    drop = OptionMenu(root, clicked, *options)
    drop.place(x=510, y=422)


    file_id_names=[]

    def upload():
        global filename
        if access == True:
            lb381 = Label(text='Time: ' + time_current + ' Hrs', bg = bgc, fg= bgc, font=('Bahnschrift Light Condensed', 38, 'bold'))
            lb381.place(x=1050, y=260) 
            t2a = time.localtime()
            time_currenta = time.strftime("%H.%M.%S", t2a)
            lb371 = Label(text='Time: ' + time_currenta + ' Hrs', bg = bgc, font=('Bahnschrift Light Condensed', 38, 'bold'))
            lb371.place(x=1050, y=260) 
            if userlogin=='0':
                print('Login First to Upload Images')
            
            elif len(filename)==0:
                print('No images Found. Capture Images First to UPLOAD')
                tk.messagebox.showerror("No Files Found ", "No images Found. Capture Images First to UPLOAD")
            else:
                tk.messagebox.showinfo("Upload Started", "Uploading of images in Progress.")
                headers = {"Authorization": "Bearer ya29.A0ARrdaM8FpvykqDRzOAETe6kutaQFNkwNtcCNMDvJeyl5wn7DXBgEJmrm8yrSTJQMHmRybV2PIPCLpYWxgl43pTBJDbSYVvTsicVXkwqIrzE1XJirj2mzgeTn15MpfKFq-nM4s0UGtV9nVRt05ErX3jkWzRMn"}
                for i in filename:
                    para = {
                        "name": i,
                        "parents": ["1kFAvDfw73mW5dc0V4bx0D3-alydFVifQ"]
                        
                    }
                    i='./Image/'+i[:]
                    files = {
                        'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
                        'file': open(i, "rb")
                    }
                    r = requests.post(
                        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                        headers = headers,
                        files = files
                    )

                    if r.text[4:9] == "error":
                        print('Request New Authorization tokens from Admin')
                        tk.messagebox.showerror("Warning!!", "Upload Failed. Request Updated Tokens from Admin")
                        break
                    else:
                        global file_id_names
                        file_id_names.append(r.text[33:66])
                        
                if len(file_id_names) >0:
                    print('succesfully uploaded ' + str(len(filename)) + ' images. Please exit application! ')
                    print("All Files Uploaded Succesfully")
                    tk.messagebox.showinfo("Success! ", "All Files Uploaded Succesfully")
                    #print(r.text[33:66]) will print file id
                    #print(r.text[4:9]) will print 'error'
                
                #EMAIL CODE HERE
                '''
                fromaddr = "arnavsumanipad@gmail.com"
                toaddr = "arnav.codesforfun@gmail.com"
                msg = MIMEMultipart()
                msg['From'] = fromaddr
                msg['To'] = toaddr
                msg['Subject'] = "ADPMS Pollution Report Ready"
                body = "Climate Change Detector (Monitoring & Awareness) user  has found pollution. Check Google Drive. Updated csv File is attached below and is now ready to be seen."
                msg.attach(MIMEText(body, 'plain'))
                filenames = "ADPMS.csv"
                attachment = open('ADPMS.csv', 'rb')
                p = MIMEBase('application', 'octet-stream')
                p.set_payload((attachment).read())
                encoders.encode_base64(p)
                p.add_header('Content-Disposition', "attachment; filename= %s" % filenames)
                msg.attach(p)
                s = smtplib.SMTP('smtp.gmail.com', 587)
                s.starttls()
                s.login(fromaddr, "Arnavcool1@")
                text = msg.as_string()
                s.sendmail(fromaddr, toaddr, text)
                s.quit()
                print('email sent')
                tk.messagebox.showinfo('Report Send', 'Report Status Updated via Email')'''
                #EMAIL CODE ENDS
                
                fcsv = open('ADPMS.csv', 'a', newline='')
                writer = csv.writer(fcsv)
                writer.writerow(['Uploaded Filenames'])
                for file in filename:
                    writer.writerow([file])
                writer.writerow("")
                fcsv.close()
                # Whatsapp CODE
                if len(file_id_names) > 0:
                    MsgBox = tk.messagebox.askquestion ('SEND WHATSAPP REPORT STATUS?', 'Do you Want to Send Report Status via Whatsapp, It may take some time! ',icon = 'warning')
                    if MsgBox == 'yes':
                        t1 = time.localtime()
                        time_now = time.strftime("%H.%M.%S", t1)
                        hour = time_now[0:2]
                        if hour[0:1] == '0':
                            hour=hour[1:]
                        hour = int(hour)
                        minute = time_now[3:5]
                        minute = int(minute) + 2
                        kit.sendwhatmsg("+919354226481","CCD (M&A) user has found pollution. Check Google Drive. Updated csv File is attached below and is now ready to be seen.",hour,minute)
                    else:
                        tk.messagebox.showinfo('Welcome Back to CCD(M&A)', 'Whatsapp Report was not Sent. Upload Process is now complete')                 
                
        else:
            tk.messagebox.showerror("Invalid Login Credentials ", "Your Login was incomplete. Log in to Upload ")    
                
    '''
    To get Authorized Tokens go to "https://developers.google.com/oauthplayground/"" --> enter scope 'https://www.googleapis.com/auth/drive' --> 
    -->Authorize APIs --> Exchange Authorization Code for Tokens--> copy access token
    '''   

        
    def exitapp():
        global filename
        MsgBox = tk.messagebox.askquestion ('Exit Application?', 'Are you sure you want to exit the application',icon = 'warning')
        if MsgBox == 'yes':
            root.destroy()
            fcsv = open('ADPMS.csv', 'a', newline='')
            writer = csv.writer(fcsv)
            tx = time.localtime()
            tx_current = time.strftime("%H.%M.%S", tx)
            if len(file_id_names) <1:
                writer.writerow(["No Files Were Uploaded"])
                writer.writerow("")
                fcsv.close()
            if len(filename) > 0:
                try:
                    for file in filename:
                        os.remove('Image/' + file)
                    print('Uploaded files removed')
                    
                except FileNotFoundError as error:
                    print('Warning! Files Deleted by User.')
                    print('Admin Updated Files were Deleted by User on purpose')
                    
                    fcsvx = open('ADPMS.csv', 'a', newline='')
                    writerx = csv.writer(fcsvx)
                    writerx.writerow(['Warning! Files were Deleted by User on purpose'])
                    fcsvx.close()
                    tk.messagebox.showerror('Warning! Files were Deleted by User on purpose. Misconduct reported to Admin.')
            
            else:
                print('ADMIN Updated that No Files were Uploaded')
            f = open('ADPMS.csv', 'a', newline='')
            w = csv.writer(f)
            w.writerow(["Log-out Time"])
            w.writerow([tx_current])
            w.writerow("")
            f.close()
        
        else:
            tk.messagebox.showinfo('Welcome Back to Climate Change Detector (M%A)', 'You will now return to the application screen')    
        
    menu = Menu(root)
    root.config(menu = menu)

    sub = Menu(menu)
    menu.add_cascade(label = "LAUNCH", menu=sub)
    sub.add_command(label = 'Login', command =login ) 
    sub.add_command(label = 'Launch Camera', command = ml )
    sub.add_command(label = 'Upload Images', command = upload )
    sub.add_command(label = 'Quit Application', command = exitapp )

    but = Button(text='Launch Webcam ', font = (None, 28), bg = btc, fg='black',command = ml)
    but.place(x = 100, y = 600)
    lb4 = Label(text='Step (2)', fg = 'black', bg = bgc, font=('Bahnschrift Light Condensed', 25, 'bold'))
    lb4.place(x = 220, y = 715)

    buta = Button(text='Upload Images', font = (None, 28), bg = btc, fg='black', command = upload)
    buta.place(x = 600, y = 600)
    lb6 = Label(text='Step (3)', fg = 'black', bg = bgc, font=('Bahnschrift Light Condensed', 25, 'bold'))
    lb6.place(x = 680, y = 715)

    but1 = Button(text = 'Quit Application', font = (None, 28), bg = btc, fg='black',command = exitapp)
    but1.place(x = 1040, y = 600) 
    lb5 = Label(text='Step (4)', fg = 'black', bg = bgc, font = ('Bahnschrift Light Condensed', 25, 'bold'))
    lb5.place(x = 1150, y = 715)

    tk.messagebox.showinfo("Log In Success!", "You are Succesfully Logged in as " + employee_fullname)

    cv2.destroyAllWindows()
    #cap.release()
    root.mainloop()
    print('')
    print('If you have any Queries you can anytime check the sahayta chatbot.')
    query = input(' Do you want to open Sahayta Chatbot? (y/n) : ')
    for i in query:
      if i =='y':
        query = 'y'
    if query.lower() == 'y':
        import nltk
        from nltk.stem.lancaster import LancasterStemmer
        stemmer = LancasterStemmer()
        import threading
        import numpy
        import tflearn
        import tensorflow as tf
        from tensorflow.python.framework import ops
        import random
        import json
        import pickle
        import os

        with open("intents.json") as file:
            data = json.load(file)

        try:
            with open("data.pickle", "rb") as f:
                words, labels, training, output = pickle.load(f)

        except:
            words = []
            labels = []
            docs_x = []
            docs_y = []

            for intent in data["intents"]:
                for pattern in intent["patterns"]:
                    wrds = nltk.word_tokenize(pattern)
                    words.extend(wrds)
                    docs_x.append(wrds)
                    docs_y.append(intent["tag"]) 

                if intent["tag"] not in labels:
                    labels.append(intent["tag"])

            words = [stemmer.stem(w.lower()) for w in words if w != "?"]
            words = sorted(list(set(words)))

            labels = sorted(labels)
            
            training = []
            output = []

            out_empty = [0 for _ in range(len(labels))]

            for x, doc in enumerate(docs_x):
                bag = []
                
                wrds = [stemmer.stem(w) for w in doc]
                
                for w in words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                output_row = out_empty[:]
                output_row[labels.index(docs_y[x])] = 1
            
                training.append(bag)
                output.append(output_row)

            training = numpy.array(training)
            output = numpy.array(output)
        
            with open("data.pickle", "wb") as f:
                pickle.dump((words, labels, training, output), f)

        ops.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)

        model = tflearn.DNN(net)

        if os.path.exists("model.tflearn.meta"):
            model.load("model.tflearn")

        else:
            model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            model.save("model.tflearn")

        def bag_of_words(s, words):
            bag = [0 for _ in range(len(words))]

            s_words = nltk.word_tokenize(s)
            s_words = [stemmer.stem(word.lower()) for word in s_words]

            for se in s_words:
                for i, w in enumerate(words):
                    if w == se:
                        bag[i] = 1

            return numpy.array(bag)

        def chat():
                
            print("Start talking with Sahayta Bot! (type quit to stop) ")
            print('')
            while True:
                inp = input("You (user) : ")
                if inp.lower() == "quit":
                    print("Sahayta Bot: thanks for visiting")
                    break

                results = model.predict([bag_of_words(inp, words)])[0]
                results_index = numpy.argmax(results)
                tag = labels[results_index]
                if results[results_index] > 0.5 :#try using 0.4)
                    for tg in data["intents"]:
                        if tg['tag'] == tag:
                            responses = tg['responses']
                            
                    print('Sahayta Bot : '+ random.choice(responses))
                    print(results[results_index])
                
                else:
                    print("Sahayta Bot: I didn't get that, try asking in a different way! ")

        chat()
    elif query.lower() == 'n':
        print('Okay, thanks for using ADPMS.')
    else:
        print('Type y or n accordingly')
    print('END OF PROGRAM ADPMS CLOSED')
    print('Thanks for visiting.')
