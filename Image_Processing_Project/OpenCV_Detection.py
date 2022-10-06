import cv2

static_back = None
prevGaussianBlur = None
diff_frame = None
count = 0;

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./video1.mp4', 0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

_, frame = cap.read()
initial_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
initial_image = cv2.GaussianBlur(initial_image, (21, 21), 0)
print("initial image ---> ",initial_image)

while True:
    _, resized = cap.read()
    # resized = cv2.resize(frame, (239, 424), interpolation=cv2.INTER_AREA)

    alpha = 5
    beta = 1.3
    diff = cv2.absdiff(frame, resized)
    diff = cv2.convertScaleAbs(diff, alpha, beta)

    isMotion = False
    isHumanDetected = False

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gaussianBlur = cv2.GaussianBlur(gray, (21, 21), 0)
    blur = cv2.blur(gaussianBlur, (5,5))

    if count%10 == 0:
        if static_back is None:
            static_back = gaussianBlur
            diff_frame = cv2.absdiff(static_back, blur)
        else:
            diff_frame = cv2.absdiff(static_back, blur)
            static_back = gaussianBlur
        count = 0
        print("entered the loop")
    count = count + 1

    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 40, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Finding contour of moving object
    cnts,_ = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if (cv2.contourArea(contour) < 5000):
            isMotion = False
            continue
        else:
            (x1, y1, w, h) = cv2.boundingRect(contour)
            # making green rectangle around the moving object
            cv2.rectangle(resized, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)
            isHumanDetected = True
            isMotion = True

    if isMotion:
        bodies = body_cascade.detectMultiScale(gray, 1.05, 4)
        #drawing for bodies
        for (x, y, width, height) in bodies:
            print("height * width", height*width)
            if (height * width >0):
                cv2.rectangle(resized, (x, y), (x + width, y + height), (255, 0, 255), 3)
                print("isMotion = true")
                isHumanDetected = True

    if isHumanDetected:
        print("human detected chance ----")
        faces = face_cascade.detectMultiScale(gray, 1.05, 4)
        # drawing faces
        for (x, y, width, height) in faces:
            print("faces---> ", faces)
            cv2.rectangle(resized, (x, y), (x + width, y + height), (255, 0, 0), 3)
            print("isHumanDetected=true")

    cv2.imshow("Camera", resized)
    cv2.imshow("threshold",thresh_frame )

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()