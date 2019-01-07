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
please refer to  [this post](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/?fbclid=IwAR04-Jlms00beR73fMR_KWza8UoIa68XlyCwT7tKKxIL4o1_oPuD6uPSqsE).
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
![image](https://user-images.githubusercontent.com/35115877/50785399-63d1a480-12a8-11e9-985b-1be1f10d8173.png)
