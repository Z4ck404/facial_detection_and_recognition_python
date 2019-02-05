# facial_detection_and_recognition_python
facial detection and recognition with python 
# install required libs : 
Anaconda is recommended here !
```
pip install opencv-python
pip install imutils 
pip install  dlib
pip install pandas 
pip install sklearn
```
if you face any issue installing dlib in windows , open anaconda prompt and type : 
```
conda install -c conda-forge dlib=19.4
```
please refer to  [this post](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/?fbclid=IwAR04-Jlms00beR73fMR_KWza8UoIa68XlyCwT7tKKxIL4o1_oPuD6uPSqsE) for more information.
# Features extraction : 
use open cv to detect 
```
def extract_feauture(img,name):
    #code here : check the file new_image.py
    shap = detect_face(img)
    #more code here : check the file new_image.py
```

##  Presnetation of the project:
This project consists in detecting the faces in a photo and see what the person identifying is in his base or not using a "classifier" which is based on the logistic regression.
##  facial detection:
This phase consists in encoding the input image in a way that will make it possible to detect the faces, among the existing methods that allow to do this we find the method
Histogram of Oriented Gradients (HOG) which consists of transforming the image into a pixel matrix in white and black and comparing the resulting shapes with a face templete generated from several faces properly processed with this method.
After we must be able to recognize the rectangles where there are faces in the photo.
For our project we also implement a pre-trained modeling that can detect in addition to face the "landmarks" that are 68 point in the face Human (see figure).
![image](https://user-images.githubusercontent.com/35115877/50806751-e3cf2d00-12ef-11e9-9b2e-764fa23e4c9c.png)
```
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
image_file = sys.argv[1]
image = cv2.imread(image_file)
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shap = face_utils.shape_to_np(shape)
    return shap
```
```
def extract_feauture(img,name):
    #img = cv2.imread(path)
    shap = detect_face(img)
    vect = {}
    d = 1
    for i in range (0,68):
        for j in range (i,68):
            a = numpy.array((shap[i][0] ,shap[i][1]))
            b = numpy.array((shap[j][0] ,shap[j][1]))
            col = "dist"+str(d)
            val = numpy.linalg.norm(a-b)
            vect[col] = val
            d = d +1
    vect["name"]= name
    return vect
 ```
 ##faces encoding  :
 This phase consists in generating for each face a vector that will represent it and help us after classifying it.
For this we generate from the 68 points detected for each face a vector which contains the Euclidean distances between the different permutations of the 68 points and by eliminating the distances d (ab) if we have already calculated the distances d (ba) that we gives a vector of 2346 measurements.
```
def extract_feauture(img,name):
    #img = cv2.imread(path)
    shap = detect_face(img)
    vect = {}
    d = 1
    for i in range (0,68):
        for j in range (i,68):
            a = numpy.array((shap[i][0] ,shap[i][1]))
            b = numpy.array((shap[j][0] ,shap[j][1]))
            col = "dist"+str(d)
            val = numpy.linalg.norm(a-b)
            vect[col] = val
            d = d +1
    vect["name"]= name
    return vect
 ```
## train a logistic regression model to cluster the data to "zakaria" and "other".
This part consists in implementing the logistic regression to have a face classifier that will make it possible to recognize Zakaria or not in the input images (the input video), but before that we apply the principal component analysis method. to reduce the drive vectors of 2346 components to a minimal number to not slow down the model and make calculations that are not necessary.
We did not give for the ACP a number of precise components to be output, but to automatically generate the number that keeps 95% of the variance.
## predict for every frame : photo in the real time video the value (zakaria or other).
test the model by typing :
 ```
 python new_image [input_image]
 #or activate uncomment the last part of the code to activate the real time face detection and recognition
 ```
  ```
  cap = cv2.VideoCapture(0)
model = train_model()
print(model)
while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        vect = extract_test(shape)
        print(predict_with_model(vect))
    
    vect = extract_test(image)
    print(predict_with_model(vect)) 
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()
  ```
 
![image](https://user-images.githubusercontent.com/35115877/50806900-88516f00-12f0-11e9-9bae-5f96b7bfd646.png)

