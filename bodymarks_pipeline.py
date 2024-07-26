import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# custom drawing styles for pose
custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
custom_connections = list(mp_pose.POSE_CONNECTIONS)

# list of landmarks to exclude from the drawing
excluded_landmarks = [
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT
]

# modify the drawing style and connections
for landmark in excluded_landmarks:
    custom_style[landmark] = DrawingSpec(color=(255, 255, 0), thickness=None)
    custom_connections = [connection_tuple for connection_tuple in custom_connections 
                            if landmark.value not in connection_tuple]

# function to draw landmarks with modified styles
def draw_custom_landmarks(image, pose_landmarks):
    mp_drawing.draw_landmarks(
        image,
        pose_landmarks,
        connections=custom_connections, # pass the modified connections list
        landmark_drawing_spec=custom_style) # and drawing style

# main program
def main():
    cap = cv2.VideoCapture(4)

    # define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror the frame

            # process the frame and get pose landmarks
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                draw_custom_landmarks(frame, results.pose_landmarks)

            # write the frame into the file 'output.avi'
            out.write(frame)

            cv2.imshow('Pose Landmarks', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # release everything when the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# run the main program
if __name__ == "__main__":
    main()
