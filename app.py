from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

def pred_image(imgData,org_img):
    url= "./static/images/pred_" + org_img.filename
    cv2.imwrite(url, imgData)  # Save the image as 'image.png' in the 'static/images' directory
    return url

@app.route('/', methods=["POST"])
def predict():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return 'No image part in the request', 400
        
        imagefile = request.files['imagefile']

        # If no file was selected
        if imagefile.filename == '':
            return 'No selected image', 400
        

        image_path = "./static/images/" + imagefile.filename
        imagefile.save(image_path)


        # yolo reads the weights and config file and creates the network.
        yolo = cv2.dnn.readNet("./static/files/yolov3.weights", "./static/files/yolov3.cfg")

        classes = []
        with open("./static/files/yolov3.txt", "r") as file:
            classes = file.read().splitlines()
        img = cv2.imread(image_path) 
        blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False) # image preprocessing
        yolo.setInput(blob)
        output_layers = yolo.getUnconnectedOutLayersNames() # extract the detection results -- names of the output layer
        outputs = yolo.forward(output_layers) # numpy array is return as output detection

        conf_threshold = 0.5 # confidence threshold
        nms_threshold = 0.4 # Non-Maximum Suppression

        boxes =[]
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * img.shape[1])
                    center_y = int(detection[1] * img.shape[0])
                    w = int(detection[2] * img.shape[1]) 
                    h = int(detection[3] * img.shape[0])
                    x= center_x - w/2 
                    y= center_y - h/2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence)) 
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold) # remove the overlapping bounding boxes

        from matplotlib import colors

        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        for i in indices:
            box = boxes[i]
            (x, y, w, h ) = box

            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (round(x),round(y)), (round(x+w),round(y+h)), color, 3)
            cv2.putText(img, label, (round(x-10),round(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return render_template('index.html', image_path=pred_image(img,imagefile))
    else :
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)