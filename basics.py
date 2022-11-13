def ml():
    if access == True:

        cap = cv2.VideoCapture(source)

        cap.set(3, 1000)
        cap.set(4, 1000)

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

            
        cap.release()
    else:
        pass