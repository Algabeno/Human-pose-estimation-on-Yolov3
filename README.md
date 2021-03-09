结合yolov3对人进行目标检测，而后对人进行姿势识别。初步实验效果发现，鲁棒性不强易受遮挡影响，但识别的准确率十分可观

# 运行步骤

1.目标检测
2.判断重心点
3.识别人体骨骼关键点

## 所需环境

torch == 1.2.0

## 运行代码
	run predict.py

## 使用视频进行预测
在predict文件里，在如下部分修改:
	

```python
cam = cv2.VideoCapture('path/vedio.mp4')
```

## 修改参数
使要显示的图片为白底图片或者原图

```python
 for n in range(person_num):
        vis_kps = np.zeros((3, joint_num))
        vis_kps[0, :] = output_pose_2d_list[n][:, 0]
        vis_kps[1, :] = output_pose_2d_list[n][:, 1]
        vis_kps[2, :] = 1
        vis_img = vis_keypoints(vis_img, vis_kps, skeleton)  # 修改要显示的图片为原图
      # vis_img = vis_keypoints(white, vis_kps, skeleton)  # 修改要显示的图片为白底图片,
```
选择保存视频的目录

```python
    vout_1.open('./output.mp4', fourcc, fps, sz, True)
```
