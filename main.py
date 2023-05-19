import cv2
from predict import detect_image

img_width = 480
img_height = 320

# Open the camera
camera = cv2.VideoCapture(0)

while (camera.isOpened()):
    # Read the frame from the camera
    ret, frame = camera.read()
    
    resized_image = cv2.resize(frame, (img_width, img_height))

    results = detect_image(resized_image)

    colors = {
        "cafe":(0,215,255),
        "jugo":(255,215,0),
        "lataverduras":(0,215,255),
        "leche":(0,215,255),
        "papas":(0,215,255),
        "refresco":(0,215,255),
    }
    price = {
        "cafe": 15,
        "jugo": 20,
        "lataverduras": 25,
        "leche":35,
        "papas":20,
        "refresco":30,
    }

    for prediction in results:
        suma = 0
        if prediction['probability'] > 0.3:
            # print(f"{prediction['tagName']}: {prediction['probability'] * 100 :.2f}%")
            suma = suma + price[prediction['tagName']]
            color = colors[prediction['tagName']]
            left = prediction['boundingBox']['left'] * img_width
            top = prediction['boundingBox']['top'] * img_height
            height = prediction['boundingBox']['height'] * img_height
            width =  prediction['boundingBox']['width'] * img_width
            cv2.rectangle(resized_image, (int(left), int(top)), (int(left + width), int(top + height)), color, 3)
            # cv2.putText(resized_image, f"{prediction['tagName']}: {prediction['probability'] * 100 :.2f}%", (int(left), int(top)-10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = color, thickness = 2)
            cv2.putText(resized_image, f"{prediction['tagName']}: {price[prediction['tagName']] :.2f}$", (int(left), int(top)-10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = color, thickness = 2)
        cv2.putText(resized_image, "{}: {}".format("Total", suma), (20, 20), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,215,255), thickness = 2)

    # Display the frame
    cv2.imshow("Frame", resized_image)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()