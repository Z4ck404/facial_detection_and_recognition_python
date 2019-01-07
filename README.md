# facial_detection_and_recognition_python
facial detection and recognition with python 
# Features extraction : 
use open cv to detect 
```
def extract_feauture(img,name):
    #code here : check the file new_image.py
    shap = detect_face(img)
    #more code here : check the file new_image.py
```
##  get mesures from the detected faces :
the previous step returns the (x,y) of 68 points in the face and this fucntion calculates the different distances between this 
points to return a vector of 2463 mesures that will be used later to train the model.

## train a logistic regression model to cluster the data to "zakaria" and "other".
## predict for every frame : photo in the rel time video the value (zakaria or other).
