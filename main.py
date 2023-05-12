import cv2
from predict import detect_image

# Open the camera
camera = cv2.VideoCapture(0)
img_width = 800
img_height = 600
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

if camera.isOpened():
    while True:
        # Read the frame from the camera
        ret, frame = camera.read()
        resized_image = cv2.resize(frame, (img_width, img_height))
        cv2.imwrite('capture.png', resized_image)

        results = detect_image('capture.png')

        colors = {
            "lleno": (0,215,255),
            "vacio": (255,215,0)
        }

        for prediction in results:
            if prediction['probability'] > 0.3:
                print(f"{prediction['tagName']}: {prediction['probability'] * 100 :.2f}%")
                color = colors[prediction['tagName']]
                left = prediction['boundingBox']['left'] * img_width
                top = prediction['boundingBox']['top'] * img_height
                height = prediction['boundingBox']['height'] * img_height
                width =  prediction['boundingBox']['width'] * img_width
                cv2.rectangle(resized_image, (int(left), int(top)), (int(left + width), int(top + height)), color, 3)
                cv2.putText(resized_image, f"{prediction['tagName']}: {prediction['probability'] * 100 :.2f}%", (int(left), int(top)-10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = color, thickness = 2)

        # Display the frame
        cv2.imshow("Frame", resized_image)

        # Check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()