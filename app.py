from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from azure.storage.blob import BlobServiceClient, __version__
import io
from io import BytesIO

app = Flask(__name__)

os.environ['AZURE_STORAGE_CONNECTION_STRING'] = 'DefaultEndpointsProtocol=https;AccountName=objectdetectionimage;AccountKey=g7t7i71TIUhgTHZ5Ql3V6IwHzdWQUHu02maWwAFG8FKj3DwUt5m3CAw0k2flwV80d/TTzxxGEvz5+AStDfsjig==;EndpointSuffix=core.windows.net'
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING') # retrieve the connection string from the environment variable
container_name = "images" # container name in which images will be saved
blob_service_client = BlobServiceClient.from_connection_string(conn_str=connect_str) # create a blob service client to interact with the storage account

try:
    container_client = blob_service_client.get_container_client(container = container_name)
    container_client.get_container_properties()
except Exception as e:
    container_client = blob_service_client.create_container(container_name)


@app.route('/')
def hello_world():
    # empty_blob_container(container_name)
    return render_template('index.html')

def empty_blob_container(container_name):
    try:
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            print(f"Deleting blob: {blob.name}")
            blob_client = blob_service_client.get_blob_client(container_name, blob.name)
            blob_client.delete_blob()
    except Exception as ex:
        print('Exception:')
        print(ex)

def pred_image(imgData,org_img):
    new_filename = "pred_" + org_img.filename

    # Encode image
    _, buffer = cv2.imencode(".jpg", imgData)

    # Convert to byte stream
    io_buf = BytesIO(buffer)
    blob_client = blob_service_client.get_blob_client(container_name, new_filename)

    try:
        blob_client.upload_blob(data=io_buf.getvalue(), blob_type="BlockBlob")
    except Exception as e:
        print(e)

    # Assuming the blob is public, construct the URL
    url = blob_client.url
    return url

def get_image_from_azure(container_name, blob_name):
    # Download the blob to a stream
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    stream = io.BytesIO()
    download_stream = blob_client.download_blob()
    download_stream.readinto(stream)

    # Convert the stream to bytes
    image_bytes = stream.getvalue()

    # Convert bytes to a numpy array
    image_nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the numpy array into an image
    image = cv2.imdecode(image_nparr, cv2.IMREAD_COLOR)

    return image

@app.route('/', methods=["POST"])
def predict():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return 'No image part in the request', 400
        
        imagefile = request.files['imagefile']

        # If no file was selected
        if imagefile.filename == '':
            return 'No selected image', 400
        
        try:
            container_client.upload_blob(imagefile.filename, imagefile)
        except Exception as e:
            print(e)

        # yolo reads the weights and config file and creates the network.
        yolo = cv2.dnn.readNet("./static/files/yolov3.weights", "./static/files/yolov3.cfg")

        classes = []
        with open("./static/files/yolov3.txt", "r") as file:
            classes = file.read().splitlines()
        # img = cv2.imread(image_path) 
        img = get_image_from_azure(container_name, imagefile.filename)
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