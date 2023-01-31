#!/usr/bin/env python
# coding: utf-8

# # Übung 2 Detektion von Verkehrsschildern

# **German Traffic Sign Detection Benchmark**
# 
# Detallierte Beschreibung des Datensatzes siehe unter folgendem [Link](http://benchmark.ini.rub.de/?section=gtsdb&subsection=news)

# ## Imports

# In[1]:


import os

import csv
import wget
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual, widgets


# In[2]:


# Testfunktion für ipywidgets: 
# Es soll ein Slider angezeigt werden. Der Wertebereich des Sliders
# soll zwischen -10(min) und 30(max) liegen. 
# Entsprechend der Sliderposition soll ein Ergebniswert angezeigt werden.
def f(x):
    return 3 * x
interact(f, x= 10);


# ## Globale Variablen
# 
# Um hartcodierte Bezeichner/Namen in den Funktionen zu vermeiden, definiere an dieser Stelle alle Variablen, die global verwendet werden.

# In[3]:


# Definiere den Pfad zum heruntergeladenen Datenordner
DATA_PATH = "C:/Users/hqpet/Desktop/BildAuto_Hausaufgabe/UE2/"

# Prüfe, ob der Pfad existiert / korrekt eingegeben wurde
assert os.path.exists(DATA_PATH), "Der angegebene Pfad existriert nicht."


# In[4]:


# Definiere den Pfad zur Datei gt.txt
"""
TIPP: arbeite zuerst mit der Datei new_gt.txt, um den Rechenaufwand bei dem Aufruf 
der Funktion calculate_hough_cirles() zu reduzieren.
Diese Datei enthält wenige Bilder mit meist gut sichtbaren Verkehrszeichen 

"""
### TO DO ###
# Definiere den Pfad zur gt.txt-Datei / zur new_gt.txt-Datei
GT_TXT_PATH = os.path.join(DATA_PATH, 'new_gt.txt')
assert os.path.exists(GT_TXT_PATH), "Der angegebene Pfad existriert nicht."


# In[5]:


# Prohibitory Class IDs 
PROHIBITORY_CLASS_IDs = [ 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]

# Mandatory Class IDs
MANDATORY_CLASS_IDs = [ 33, 34, 35, 36, 37, 38, 39, 40 ]

# Danger Class IDs
DANGER_CLASS_IDs = [ 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ]


# In[6]:


# MANDATORY-Dictionary 
MANDATORY_DICT = {}
# MANDATORY images (filenames) list
MANDATORY_IMG_LIST = []
# MANDATORY filepaths
MANDATORY_FILEPATHS = []


# In[7]:


def calculate_mandatory(): 
    """
    Sortiert Verkehrszeichen nach der Kategorie „mandatory“ und speichert 
    die Ergebnisse in ein Dictionary.
    Das Dictionary beinhaltet Dateinamen als Schlüssel und Listen von 
    Ground Thruth ROIs-Listen als Values.
    """
    global MANDATORY_DICT
    # Setze prohibitory dictionary zurück
    MANDATORY_DICT.clear()
    
    # Öffne die gt.txt-Datei / new_gt.txt-Datei
    with open(GT_TXT_PATH, newline='') as csvfile:
        gt_reader = csv.reader(csvfile, delimiter=';')#gt_reader中存储的是列表类型数据，为了方便理解我们就将gt_reader看成是个列表嵌套列表的对象
        
        # Bau eine Schleife, um die Daten Zeile für Zeile auszulesen 
        # und die entsprechende Liste der ROIs für die Datei zu füllen
        for row in gt_reader: #row是一个列表
            ### TO DO ###
            key = str(row[0])
            if key not in MANDATORY_DICT:
                 MANDATORY_DICT[key] = []
            MANDATORY_DICT[key].append(list(map(int,row[1:6])))


# ## Aufgabe 1 - Aussortieren bestimmter Verkehrsschilder

# In[8]:


# Ermittele die Dateienamen (ausgehend von DATA_PATH) alle Treffer in MANDATORY_DICT
# MANDATORY_FILEPATHS = os.path.splitext(DATA_PATH)[0]
MANDATORY_FILEPATHS = [os.path.join(DATA_PATH, 'FullIJCNN2013', key) for key in MANDATORY_DICT.keys()]
print(len(MANDATORY_FILEPATHS))


# In[9]:


# Funktionsaufruf
"""
Erwartete Ausgabe:
    {'00002.ppm': [[892, 476, 1006, 592, 39]],
     '00039.ppm': [[953, 252, 1015, 313, 35]],
     '00040.ppm': [[1040, 310, 1130, 400, 33]],
     '00066.ppm': [[569, 483, 618, 535, 38]],
     '00074.ppm': [[383, 527, 421, 569, 38]],
     '00081.ppm': [[233, 499, 249, 521, 38]],
     ...}
"""
calculate_mandatory()
print(MANDATORY_DICT)


# In[10]:


def render_mandatory_rois():
    """
    Malt die ROIs (Rechtecke) auf die entsprechenden Bilder und speichert 
    die in MANDATORY_IMG_LIST. 
    Hinweis:
    Die ROIs und Bildernamen können aus MANDATORY_DICT ermittelt werden
    """
    # Setze die globale variable zurück
    MANDATORY_IMG_LIST.clear()
    
    for key in MANDATORY_DICT.keys():
        file_path = os.path.join(DATA_PATH,"FullIJCNN2013", key)
        img = plt.imread(file_path)
        ### TO DO ###
        for idx in range(len(MANDATORY_DICT[key])):
            # Berechne Koordinaten des Rechtecks
            point1 = (MANDATORY_DICT[key][idx][0],MANDATORY_DICT[key][idx][1])
            point2 = (MANDATORY_DICT[key][idx][2],MANDATORY_DICT[key][idx][3])
            
            # Zeichne das Rechteck
            img = cv2.rectangle(img,point1,point2,(0,255,0),2)
            org =  (MANDATORY_DICT[key][idx][2] + 10,MANDATORY_DICT[key][idx][1]+20)          
            # Speichere Verkehrszeichennamen als text
            text = str(MANDATORY_DICT[key][-1])
            img = cv2.putText(img,text,org,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),5)
            
        MANDATORY_IMG_LIST.append(img)


# In[11]:


# Mandatory Image list abrufen
render_mandatory_rois()
print(len(MANDATORY_IMG_LIST))
# print(MANDATORY_IMG_LIST)


# In[12]:


def show_img(idx):
    plt.figure(figsize=(16,8))
    plt.imshow(MANDATORY_IMG_LIST[idx])
    plt.show()


# In[13]:


interact(show_img, idx=widgets.IntSlider(min=0,max=len(MANDATORY_IMG_LIST)-1, step=1, value=0));


# ## Aufgabe 2 – Formbasierter Ansatz
# 
# 
# Hier kannst du die Links abrufen, die zum Implementieren dieser Funktion nützlich sein können:
# 
# - https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
# - https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# - https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html 
# 
# Übersicht der Tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

# In[14]:


# import shutil
# for file_path in MANDATORY_FILEPATHS:
#     new_file_path = "C:/Users/hqpet/Desktop/BildAuto_Hausaufgabe/UE2/FullIJCNN2013" + os.path.split(file_path)[1]
#     shutil.copyfile(file_path, new_file_path)


# In[15]:


def calculate_hough_cirles(filepaths, d_p, min_dist, param1, param2, min_radius, max_radius):
    """
     Berechnet Hough Circles unter Berücksichtigung der Form der Verkehrszeichen
    """
    
    # Liste fuer die Speicherung des Ergebnis
    result = []
    predicted_dict = {}
    
    for filepath_ in filepaths:
        # Lade das Bild in color_img
        color_img = cv2.imread(filepath_, cv2.IMREAD_COLOR)
        ### TO DO ###
        
        # Konvertiere das BGR-Bild in Gray.
        # https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
        img_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # Reduziere das Rauschen 
        # https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html --> Kapitel "Image bluring"
        img_blurred = cv2.blur(img_gray,(2,2))
        
        # Ermittele die Kreisen auf dem Bild 
        # https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
        circles = cv2.HoughCircles(img_blurred,cv2.HOUGH_GRADIENT, dp=d_p, minDist=min_dist, 
                                  param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

        # Kreise auf das Bild malen
        if circles is not None: 
            # Kreise-Paramater in interger umwandeln
            circles = np.uint16(np.around(circles)) 
            
            # Kreise auf das Bild malen
            for point in circles[0, :]: 
                a, b, r = point[0], point[1], point[2]#color_img.shape
                cv2.circle(img_blurred,(a,b),r,(0,255,0),2)
                # Ermittle Koordinaten der Rechtecke, die für die Evaluation benutzt werden
                point1 = a-r,b-r
                point2 = a+r,b+r
                # OPTIONAL: Rechtecke auf das Bild malen
                cv2.rectangle(color_img,point1,point2,(0,255,0),2)
                if os.path.split(filepath_)[-1] in predicted_dict:
                    if a==0 and b==0 and r==0:
                        continue
                    predicted_dict[os.path.split(filepath_)[-1]].append([point1[0], point1[1], point2[0], point2[1]])
                else: 
                    predicted_dict[os.path.split(filepath_)[-1]] = []
                    if a==0 and b==0 and r==0:
                        continue
                    predicted_dict[os.path.split(filepath_)[-1]].append([point1[0], point1[1], point2[0], point2[1]])
        else:
            predicted_dict[os.path.split(filepath_)[-1]] = []                
        result.append(color_img)
        
    return result, predicted_dict


# In[19]:

#
# pred_imgs_form, predicted_rect_rois = calculate_hough_cirles(MANDATORY_FILEPATHS,
#                                                   d_p=1,
#                                                   min_dist=120,
#                                                   param1= 100,
#                                                   param2= 1,
#                                                   min_radius=5,
#                                                   max_radius=50)


# In[20]:


def show_img_form(idx):
    plt.figure(figsize=(16,8))
    plt.imshow(cv2.cvtColor(pred_imgs_form[idx], cv2.COLOR_BGR2RGB))
    plt.show()


# In[21]:

#
# interact(show_img_form, idx=widgets.IntSlider(min=0, max=len(pred_imgs_form)-1, step=1, value=0));


# ## Aufgabe 3 – Optimierung und Evaluation des formbasierten Ansatzes 对基于形式的方法进行优化和评估

# In[22]:


def jaccard_similarity(pred, gr_truth):
    '''
    Berechnet den Jaccard-Koeffizienten für zwei Rechtecke: den vorhergesagen (pred) und den ground_truth (gr_truth)
    
    '''
    # Ermittle die (x, y)-Koordinaten der Schnittmenge beider Rechtecke
    x_i1 = max(pred[0], gr_truth[0])
    y_i1 = max(pred[1], gr_truth[1])
    x_i2 = min(pred[2], gr_truth[2])
    y_i2 = min(pred[3], gr_truth[3])

    inter_area = max(0, x_i2 - x_i1 + 1) * max(0, y_i2 - y_i1 + 1)

    pred_area = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    gr_truth_area = (gr_truth[2] - gr_truth[0] + 1) * (gr_truth[3] - gr_truth[1] + 1)
    
    iou = inter_area / float(pred_area + gr_truth_area - inter_area)
    
    # Gebe den "Intersection Over Union"-Wert zurück
    return iou


# In[29]:


def evaluate_detection(ground_truth_dict, predicted_dict, similarity_threshold=0.6):    
    '''
    Evaluiert implementierte Ansätze anhand des Jaccard-Ähnlichkeitsmaßes
    Referenz für die Berechnung: Houben et. al. Kapitel IV Evaluation Procedure
    '''
    # True Positives
    tp = 0
    # False Positives
    fp = 0
    # False Negatives
    fn = 0
    
    for key in ground_truth_dict.keys():
        # Liste mit allen ROIs eines Dateinamens 
        rois_gt_lists = ground_truth_dict[key]
        
        # Berechne Jaccard-Ähnlichkeitsmaß von detektierten Rechtecken, die aus den Kreiskoordinaten ermittelt wurden
        rois_pred_lists = predicted_dict[key]
        
        if len(rois_pred_lists) > 0:
            for rois_gt_list in rois_gt_lists:
                iou = [jaccard_similarity(rois_pred, rois_gt_list) for rois_pred in rois_pred_lists]
                
                # Liste mit den Werten, die kleiner als similarity_threshold sind
                iou_lt_threshold = [value for value in iou if value < similarity_threshold]
                fp = fp + len(iou_lt_threshold)
                
                # Liste mit den Werten, die größer / gleich similarity_threshold sind
                iou_gt_threshold = [value for value in iou if value >= similarity_threshold]
                
                if len(iou_gt_threshold) > 0 : 
                    tp = tp + 1
                else:
                    fn = fn + 1
        else:
            fn = fn + len(rois_gt_lists)

    return tp, fp, fn            


# In[30]:


def calculate_precision_recall(tp, fp, fn):
    '''
    Berechnet Precision- und Recall-Werte
    '''
    
#     precision = math.nan
    precision = float('nan')
    if tp + fp != 0:
        precision = tp / (tp + fp)   
    
#     recall = math.nan
    recall = float('nan')
    if tp + fn != 0:
        recall = tp / (tp + fn)
        
    return precision, recall


# In[31]:


# similarity_threshold entspricht dem Schwellenwert im Paper von Houben et. al. Similarity_threshold对应于Houben论文中的阈值。
# tp_form, fp_form, fn_form = evaluate_detection(MANDATORY_DICT, predicted_rect_rois, similarity_threshold=0.6)


# In[32]:


# Precision-Recall-Plot
### TO DO ###


# In[33]:


precision_list = []
recall_list = []
for value in range(100,40,-2):
    pred_imgs_form,predicted_rect_rois = calculate_hough_cirles(MANDATORY_FILEPATHS,
                                                                d_p=1,
                                                                min_dist =120,
                                                                param1=100,
                                                                param2=value,
                                                                min_radius=5,
                                                                max_radius = 100)
    tp_form,fp_form,fn_form = evaluate_detection(MANDATORY_DICT,predicted_rect_rois,similarity_threshold=0.6)
    precision,recall =calculate_precision_recall(tp_form,fp_form,fn_form)
    precision_list.append(precision)
    recall_list.append(recall)
    
plt.figure()
plt.plot(recall_list,precision_list,marker='.',label='Hough Circle')
plt.title('PR-curve')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.show()


