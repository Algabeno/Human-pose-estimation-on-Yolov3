import pickle
from score import Score
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import cv2
import posenet

def get_adjacent_keypoints(keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        results.append(
            # np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
            np.array([keypoint_coords[right][::-1], keypoint_coords[left][::-1]]).astype(np.int32),
        )
    return results

def draw_skel_and_kp(flag,
        img, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    ii = 0
    ks = 0.99
    new_keypoints = get_adjacent_keypoints( keypoint_coords[ii, :, :], min_part_score)
    adjacent_keypoints.extend(new_keypoints)

    for kc in keypoint_coords[ii, :, :]:
        cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
        # cv_keypoints.append(cv2.KeyPoint(kc[0], kc[1], 10. * ks))


    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",cv_keypoints)
    if flag==1:

        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=True, color=(255, 255, 255))
        return out_img
    elif flag==0:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 0, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=True, color=(0, 255, 0))
        return out_img

def main():
    s = Score()
    i = 0
    j = 0
    # data1 = pickle.load(open("cgy1.pickle", 'rb')) #96
    # cap = cv2.VideoCapture("cgy1.mp4")

    data1 = pickle.load(open("ball_test01.pickle", 'rb')) #96
    cap = cv2.VideoCapture("ball_test01.mp4")

    for (k, v) in data1.items():
         if k == "data":
              (data1, i) = (v, v.shape[0])

    # black_image = np.zeros((372, 495, 3), dtype='uint8')
    print(data1.shape)

    if cap.isOpened() is False:
        print("error in opening video")

    for num in range(data1.shape[0]):
        ret_val, image = cap.read()
        image = cv2.resize(image, (372, 495))
        keypoint_coords =  data1[num:num+1, :,:]
        pos_temp_data = []
        for c in keypoint_coords[0, :, :]:
            pos_temp_data.append(c[1])
            pos_temp_data.append(c[0])

        pos_temp_data = np.asarray(pos_temp_data).reshape(1, 17, 2)
        image = draw_skel_and_kp(1, image,  pos_temp_data,
                        min_pose_score=0.1, min_part_score=0.0001)
        cv2.imshow("image", image)
        cv2.waitKey(30)

if __name__ == "__main__":
    main()


