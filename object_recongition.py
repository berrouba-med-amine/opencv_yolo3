import cv2

# Load the YOLOv3 network
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load the class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define the input image
img = cv2.imread('image.jpg')

# Get the image dimensions and create a blob
height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input for the network and make a forward pass
net.setInput(blob)
layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(layer_names)

# Define some lists to store the detected objects
boxes = []
confidences = []
class_ids = []


# Loop over each output layer
for output in outputs:
    # Loop over each detection
    for detection in output:
        # Get the class probabilities and find the index of the class with the highest probability
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Calculate the center, width, and height of the bounding box
            cx = int(detection[0] * width)
            cy = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Calculate the top-left corner of the bounding box
            x = int(cx - w/2)
            y = int(cy - h/2)
            
            # Store the bounding box information
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Loop over the indices and draw the bounding boxes and labels on the image
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = classes[class_ids[i]]
    confidence = confidences[i]
    
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, f'{label} {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the output image
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
