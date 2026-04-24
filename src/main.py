import cv2
import os
import mediapipe as mp

from pose_estimation import PoseEstimator
from angle_calculation import calculate_angle
from gait_features import GaitFeatures
from csv_writer import CSVWriter

mp_pose = mp.solutions.pose

# 🔥 CHANGE THIS ONLY
MODE = "side"   # "side" or "front"

VIDEO_PATH = "../data/raw_videos/Walking1.mp4"
CSV_PATH = "../results/gait_data.csv"

cap = cv2.VideoCapture(VIDEO_PATH)

pose_estimator = PoseEstimator()
features = GaitFeatures(mode=MODE)

# CSV headers
if MODE == "side":
    csv_writer = CSVWriter(CSV_PATH,
        ["frame", "left_knee", "right_knee", "left_hip", "right_hip"])
else:
    csv_writer = CSVWriter(CSV_PATH,
        ["frame", "step_width", "hip_sway", "shoulder_sway"])

frame_idx = 0

def get_point(lm):
    return [lm.x, lm.y], lm.visibility

def get_xy(lm):
    return lm.x, lm.y, lm.visibility

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = pose_estimator.process_frame(frame)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        if MODE == "side":
            # LEFT
            l_hip, lhv = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
            l_knee, lkv = get_point(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
            l_ankle, lav = get_point(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
            l_sh, lsv = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

            # RIGHT
            r_hip, rhv = get_point(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
            r_knee, rkv = get_point(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            r_ankle, rav = get_point(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            r_sh, rsv = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

            if min(lhv, lkv, lav, lsv, rhv, rkv, rav, rsv) > 0.5:
                lk = calculate_angle(l_hip, l_knee, l_ankle)
                lh = calculate_angle(l_sh, l_hip, l_knee)

                rk = calculate_angle(r_hip, r_knee, r_ankle)
                rh = calculate_angle(r_sh, r_hip, r_knee)

                features.add_side(lk, rk, lh, rh)

                csv_writer.write_row([frame_idx, lk, rk, lh, rh])

        else:  # FRONT MODE
            lx, _, lv = get_xy(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
            rx, _, rv = get_xy(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

            lhip_x, _, lhv = get_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
            rhip_x, _, rhv = get_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])

            lsh_x, _, lsv = get_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            rsh_x, _, rsv = get_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

            if min(lv, rv, lhv, rhv, lsv, rsv) > 0.5:
                step_width = abs(lx - rx)
                hip_dev = abs(lhip_x - rhip_x)
                shoulder_dev = abs(lsh_x - rsh_x)

                features.add_front(step_width, hip_dev, shoulder_dev)

                csv_writer.write_row([frame_idx, step_width, hip_dev, shoulder_dev])

    frame = pose_estimator.draw_landmarks(frame, result)
    cv2.imshow("Hybrid Gait System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
csv_writer.close()

print("Final Features:")
print(features.get_features())
print(f"CSV saved at: {CSV_PATH}")