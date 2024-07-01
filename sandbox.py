import cv2
import mediapipe as mp
import numpy as np

# Constants for face tracking
MAX_STABLE_FRAMES = 30  # Maximum consecutive stable frames to consider a face as real
DISTANCE_THRESHOLD = (
    10  # Maximum Euclidean distance between landmarks for a stable face
)
MOTION_THRESHOLD = 20  # Minimum motion required to consider a face as real


def is_real_face(prev_landmarks, current_landmarks, stable_frames, motion_threshold):
    if not prev_landmarks or not current_landmarks:
        return False

    distances = [
        np.linalg.norm(np.array(prev) - np.array(curr))
        for prev, curr in zip(prev_landmarks, current_landmarks)
    ]

    is_stable = all(d < DISTANCE_THRESHOLD for d in distances)
    has_motion = sum(distances) > motion_threshold

    return is_stable and has_motion and stable_frames >= MAX_STABLE_FRAMES


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    prev_detected_faces = []  # To store previously detected faces
    prev_face_landmarks = []  # To store previously detected face landmarks
    stable_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(
            frame, 1
        )  # Flip the frame horizontally for a more natural view

        results_detection = face_detection.process(frame)
        detected_faces = []  # To store currently detected faces
        detected_landmarks = []  # To store currently detected landmarks

        if results_detection.detections:
            for detection in results_detection.detections:
                if bounding_box := detection.location_data.relative_bounding_box:
                    ih, iw, _ = frame.shape
                    x, y, w, h = (
                        int(bounding_box.xmin * iw),
                        int(bounding_box.ymin * ih),
                        int(bounding_box.width * iw),
                        int(bounding_box.height * ih),
                    )
                    detected_faces.append((x, y, x + w, y + h))
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )  # Draw bounding box

        for x1, y1, x2, y2 in detected_faces:
            face_frame = frame[y1:y2, x1:x2].copy()

            if face_frame.size != 0:
                face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                results_mesh = face_mesh.process(face_frame_rgb)

                face_landmark = []  # To store landmarks for the current face

                if results_mesh.multi_face_landmarks:
                    for landmarks in results_mesh.multi_face_landmarks:
                        for landmark in landmarks.landmark:
                            x, y = (
                                int(landmark.x * (x2 - x1)) + x1,
                                int(landmark.y * (y2 - y1)) + y1,
                            )
                            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                            face_landmark.append((x, y))

                if is_real_face(
                    prev_face_landmarks, face_landmark, stable_frames, MOTION_THRESHOLD
                ):
                    detected_landmarks.append(face_landmark)

        # Draw labels for each detected face
        for i, (x1, y1, x2, y2) in enumerate(detected_faces):
            label_position = (x1, y1 - 10)
            cv2.putText(
                frame,
                f"Face {i + 1}",
                label_position,
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        if (
            detected_faces != prev_detected_faces
            or detected_landmarks != prev_face_landmarks
        ):
            stable_frames += 1
        else:
            stable_frames = 0

        cv2.imshow("Face Detection and Landmarking", frame)
        prev_detected_faces = detected_faces[:]
        prev_face_landmarks = detected_landmarks[:]

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
