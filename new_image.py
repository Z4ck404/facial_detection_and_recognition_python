from imutils import face_utils
import dlib
import cv2
import numpy
import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# default solver is incredibly slow which is why it was changed to 'lbfgs'.
logisticRegr = LogisticRegression(solver = 'lbfgs')
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
#construct your numpy array of data
image_file = sys.argv[1]
image = cv2.imread(image_file)
#the function to extract  the 2346 features from a photo :
#the pattern I ll be using to store the data :
#data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
#df = pd.DataFrame(data)
#print df
def take_picture(s):
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            cam.release()
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = ""+s+".png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
### to load image from a file :
#take_picture("new0")
#image = cv2.imread('new0.png')
def load_image(folder_name):
    path = folder_name + '/'+str(1)+'.jpg'
    #path = 'zakaria/14.jpg'
    image = cv2.imread(path)
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def extract_test(img):
    #img = cv2.imread(path)
    shap = detect_face(img)
    vect = {}
    d = 1
    for i in range (0,68):
        for j in range (i,68):
            a = numpy.array((shap[i][0] ,shap[i][1]))
            b = numpy.array((shap[j][0] ,shap[j][1]))
            #names = names + ["dist"+str(d)]
            #distances=distances+[numpy.linalg.norm(a-b)] 
            col = "dist"+str(d)
            val = numpy.linalg.norm(a-b)
            vect[col] = val
            d = d +1
    #return vect
    ve = pd.DataFrame([vect])
    #ve = StandardScaler().fit_transform(ve)
    #pca = PCA(n_components=100)
    #pca = PCA(.95)
    #principalComponents = pca.fit_transform(ve)
    #rst = pd.DataFrame(principalComponents)
    return ve
def extract_feauture(img,name):
    #img = cv2.imread(path)
    shap = detect_face(img)
    vect = {}
    d = 1
    for i in range (0,68):
        for j in range (i,68):
            a = numpy.array((shap[i][0] ,shap[i][1]))
            b = numpy.array((shap[j][0] ,shap[j][1]))
            #names = names + ["dist"+str(d)]
            #distances=distances+[numpy.linalg.norm(a-b)] 
            col = "dist"+str(d)
            val = numpy.linalg.norm(a-b)
            vect[col] = val
            d = d +1
    vect["name"]= name
    return vect
#check all the photos in the folder to train the model : 
#check the folder folder_name and extract data from its photo
#n = number of photos in the folder .
def train(folder_name,n):
    data = []
    for i in range (1,n+1):
        path = folder_name+"/"+str(i)+".jpg"
        image = cv2.imread(path)
        v = extract_feauture(image,folder_name)
        data = data + [v]
    return data
#gather all the extracted data from photos in one data frame : 
def train_model():
    v1 = train('zakaria',16)
    v2 = train('other',4)
    data = v1+v2
    model = pd.DataFrame(data)
    #features = list(model.columns.values)[:2346]
    #x = model.loc[:, features].values
    #y = model.loc[:,['name']].values
    #x = StandardScaler().fit_transform(x)
    #pca = PCA(n_components=100)
    #principalComponents = pca.fit_transform(x)
    #rst = pd.DataFrame(principalComponents)
    #finalDf = pd.concat([rst, model[['name']]], axis = 1)
    return model
#apply logistic regression and predict who is in the picture :
def predict_with_model(vect):
    model = train_model()
    features = list(model.columns.values)[:2346]
    x = model.loc[:, features].values
    y = model.loc[:,['name']].values
    logisticRegr.fit(x, y)
    return logisticRegr.predict(vect)
def detect_face(image):
    #image = cv2.imread('new0.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shap = face_utils.shape_to_np(shape)
    return shap
#apply pca to reduce dimentionality 
def apply_pca(shape):
    pca = PCA(.95)
    pca.fit(shape)
    return  pca.transform(shape)
    #print(PCA(shape))  
def get_mesures(shape):
    shap = apply_pca(shape)
    mes_vet = []
    for i in range (0,68):
        for j in range (i,68):
            a = numpy.array((shap[i][0] ,shap[i][1]))
            b = numpy.array((shap[j][0] ,shap[j][1]))
            mes_vet=mes_vet+ [numpy.array((0, numpy.linalg.norm(a-b)))]
    return mes_vet
def draw_img(img,shape):
    for (x, y) in shape:
        image = cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Output", image)
            #k = cv2.waitKey(5) & 0xFF
#s = detect_face(image)
#tab = extract_feauture(image)
#print(tab)
#load_image("zakaria")
#for i in tab :
#    print(i)
vect = extract_test(image)
print (vect)
#print(train_model())
print(predict_with_model(vect))
#print(train_model())
#path = "/zakaria/1.jpg"
#img = cv2.imread(path)
#extract_feauture(img)
#tab = numpy.asarray(tab)
#tab = tab.reshape(1,-1)
#print(len(tab))
#print("len data with acp ",len(apply_pca(tab)))
#print("len data without acp",len(get_mesures(s)))
#print(apply_pca(s)[0][0])
#print("len de data without pcs : ",len(s)," ",len(s[0]))
#print("len de data after applying pca:",len(apply_pca(s)))


