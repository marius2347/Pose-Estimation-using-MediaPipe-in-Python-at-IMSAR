import cv2
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# initialize Pose model
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def draw_landmarks_with_confidence(image, pose_landmarks):
    confidence_data = []
    if pose_landmarks:
        for landmark in pose_landmarks.landmark:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            confidence = landmark.visibility
            # draw landmark
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            # draw confidence score
            cv2.putText(image, f'{confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # collect the confidence scores
            confidence_data.append((x, y, confidence))
    return confidence_data

def main():
    cap = cv2.VideoCapture(4) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_with_confidence.avi', fourcc, 20.0, (640, 480))

    # ppen a CSV file to save confidence scores
    with open('confidence_scores.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'Landmark X', 'Landmark Y', 'Confidence'])

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            frame = cv2.flip(frame, 1)  # Mirror the frame
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # draw landmarks and collect confidence scores
            confidence_data = draw_landmarks_with_confidence(frame, results.pose_landmarks)
            if confidence_data:
                for x, y, confidence in confidence_data:
                    csv_writer.writerow([frame_number, x, y, confidence])

            out.write(frame)
            cv2.imshow('Pose Landmarks with Confidence', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
