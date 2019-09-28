from xml.dom.minidom import Document
import os
import os.path
import xml.etree.ElementTree as ET
from tqdm import tqdm
opj = os.path.join
  
txt_path = "submit.txt"
xml_path = "inference_xml"
img_name=[]
 
if not os.path.exists(xml_path):
    os.mkdir(xml_path)
 
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
 
def txttoxml(txtPath,xmlPath):
    dict = {'1':"car",
            '2':"person",
            '3':"truck",
            '4':"bus",
            '5':"rider",
            '6':"rear",
            '7':"front"}
    txtFile = open(txtPath)
    txtList = txtFile.readlines()
    for i in tqdm(txtList):
        oneline = i.strip().split(" ")
        if oneline[0] not in img_name:
            img_name.append(oneline[0])
             
            xmlBuilder = Document()
            annotation = xmlBuilder.createElement("annotation")
            xmlBuilder.appendChild(annotation)
            filename = xmlBuilder.createElement("filename")
            filenameContent = xmlBuilder.createTextNode(oneline[0]+".jpg")
            filename.appendChild(filenameContent)
            annotation.appendChild(filename)
            f = open(opj(xmlPath, oneline[0]+".xml"), 'w')
            xmlBuilder.writexml(f, newl='\n', addindent='  ')
            f.close()
         
        tree = ET.parse(opj(xmlPath, oneline[0]+".xml"))
        root = tree.getroot()
         
        obj = ET.Element("object")
        name = ET.Element("name")
        name.text = dict[oneline[1]]
        obj.append(name)
         
        bndbox = ET.Element("bndbox")
        xmin = ET.Element("xmin")
        xmin.text = oneline[2]
        bndbox.append(xmin)
        ymin = ET.Element("ymin")
        ymin.text = oneline[3]
        bndbox.append(ymin)
        xmax = ET.Element("xmax")
        xmax.text = str(int(oneline[2])+int(oneline[4]))
        bndbox.append(xmax)
        ymax = ET.Element("ymax")
        ymax.text = str(int(oneline[3])+int(oneline[5]))
        bndbox.append(ymax)
         
        obj.append(bndbox)
        root.append(obj)
        indent(root)
        tree.write(opj(xmlPath, oneline[0]+".xml"))
 
txttoxml(txt_path,xml_path)
