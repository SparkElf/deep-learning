import numpy as np
import json
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


def landmark_json2mp4(src, dest):

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing_styles = mp.solutions.drawing_styles

    with open(src) as fp:
        data = json.load(fp)

    writer = cv2.VideoWriter(
        dest, cv2.VideoWriter_fourcc(*'XVID'), 30., (640, 480))

    for landmarks_list in data:
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in landmarks_list:
            m = landmark_pb2.NormalizedLandmark()
            m.x = x
            m.y = y
            m.z = z
            landmarks.landmark.append(m)
        landmarks.landmark.append(m)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

        image = cv2.flip(image, 1)
        writer.write(image)
    writer.release()
