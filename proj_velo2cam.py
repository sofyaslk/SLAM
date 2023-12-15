import sys
import cv2
import numpy as np
import os
from scipy.optimize import least_squares
import copy
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

focal_length = 718.8560
pp = (607.1928, 185.2157)

K = np.array([[focal_length, 0, pp[0]],
      [0, focal_length, pp[1]],
      [0, 0, 1]])

def find_features(img0, img1):
    img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(img0gray, None)
    kp1, des1 = sift.detectAndCompute(img1gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

    return pts0, pts1

def triagulate_image_data(P1, P2, pts1, pts2, K, repeat):
    if not repeat:
        points1 = np.transpose(pts1)
        points2 = np.transpose(pts2)
    else:
        points1 = pts1
        points2 = pts2

    cloud = cv2.triangulatePoints(P1, P2, points1, points2)
    cloud = cloud / cloud[3]
    return points1, points2, cloud
    
def get_visual_depth(cloud):
    # Extract the Z-coordinates from the triangulated points
    visualdep = cloud[2]

    return visualdep

def PnP(X, p, K, d, p_0, initial):
    # print(X.shape, p.shape, p_0.shape)
    if initial == 1:
        X = X[:, 0, :]
        p = p.T
        p_0 = p_0.T

    ret, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvecs)

    if inliers is not None:
        p = p[inliers[:, 0]]
        X = X[inliers[:, 0]]
        p_0 = p_0[inliers[:, 0]]

    return R, t, p, X, p_0

def get_velodyne_data(binary, calib):
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1, 4))   
    # Extract Lidar points
    points = scan[:, 0:3]
    # Remove points behind the Lidar sensor (x-coordinate < 0)
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)

    # Remove points with negative depth (z-coordinate < 0)
    #velo = np.delete(velo, np.where(velo[2, :] < 0), axis=1)
    return velo
    
    
"""
def get_velodyne_data(binary, calib):
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    print('##P2', P2)
    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]  # lidar xyz (front, left, up)
    # TODO: use fov filter?
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))
    cam = np.delete(cam, np.where(cam[2, :] < 0), axis=1)
    # cam[:2] /= cam[2, :]   
    return cam
   
def transform_point_to_camera(X, Tr, K):
    # Convert the input point to homogeneous coordinates
    X_homogeneous = np.append(X, 1)

    # Perform the transformation: x = K * Tr * X
    x_homogeneous = K.dot(Tr.dot(X_homogeneous))

    # Normalize homogeneous coordinates
    if len(x_homogeneous) >= 4 and x_homogeneous[3] != 0:
        x = x_homogeneous[:3] / x_homogeneous[3]
    else:
        # Handle the case where there is no fourth element or it's zero
        x = x_homogeneous[:3]
    return x
def get_velodyne_data(binary):

    Tr_matrix = np.array([[4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
                         [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
                         [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01]])
    # Load Velodyne binary data
    data = np.fromfile(binary, dtype=np.float32).reshape(-1, 4)
    # Extract x, y, z coordinates
    points_velodyne = data[:, :3]
    # Transform points to left rectified camera coordinates
    points_camera = np.apply_along_axis(lambda x: transform_point_to_camera(x, Tr_matrix, K), axis=1, arr=points_velodyne)
    return points_camera
"""

def superimposed_lidar_image(name, binary, calib):
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
    points = scan[:, 0:3] # lidar xyz (front, left, up)
    # TODO: use fov filter? 
    velo = np.insert(points,3,1,axis=1).T
    velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
    # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
    png = mpimg.imread(img)
    if len(png.shape) == 3:  # Color image
        IMG_H, IMG_W, _ = png.shape
    else:  # Grayscale image
        IMG_H, IMG_W = png.shape
        png = np.stack((png,) * 3, axis=-1) 
    # restrict canvas in range
    plt.axis([0,IMG_W,IMG_H,0])
    plt.imshow(png)
    # filter point out of canvas
    u,v,z = cam
    print(z)
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)
    # generate color map from depth
    u,v,z = cam
    plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
    image = f'D:\\SOFYA\\dataset\\sequences\\00\\projection\\{name}.png'
    plt.savefig(image, bbox_inches='tight')
    return image


########MAIN########

R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P1 = np.matmul(K, R_t_0)
P2 = np.empty((3, 4))
R_t_1 = np.empty((3, 4))

for sn in range(1):
    name = '%06d' % sn  # 6 digit zeropadding
    img = f'D:\\SOFYA\\dataset\\sequences\\00\\seq\\{name}.png'
    next_img = sn + 1
    img1 = f'D:\\SOFYA\\dataset\\sequences\\00\\seq\\{next_img:06d}.png'
    binary = f'D:\\SOFYA\\dataset\\sequences\\00\\velodyne_debug\\{name}.bin'
    with open(f'D:\\SOFYA\\Data object KITTI\\testing\\calib\\{name}.txt', 'r') as f:
        calib = f.readlines()
    if sn < 2:
        img0 = cv2.imread(img)
        img1 = cv2.imread(img1)
        pts0, pts1 = find_features(img0, img1)

        # Finding essential matrix
        E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
        pts0 = pts0[mask.ravel() == 1]
        pts1 = pts1[mask.ravel() == 1]
        # The pose obtained is for second image with respect to first image
        _, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)  # |finding the pose
        pts0 = pts0[mask.ravel() > 0]
        pts1 = pts1[mask.ravel() > 0]
        R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
        R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())
        P2 = np.matmul(K, R_t_1)

        pts0, pts1, points_3d = triagulate_image_data(P1, P2, pts0, pts1, K, repeat=False)
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
        #points_3d = points_3d[:, 0, :]
        Rot, trans, pts1, points_3d, pts0t = PnP(points_3d, pts1, K, np.zeros((5, 1), dtype=np.float32), pts0, initial=1)
        
    else:
        img2 = cv2.imread(img)
        pts_, pts2 = find_features(img1, img2)
        pts0, pts1, points_3d = triagulate_image_data(P1, P2, pts0, pts1, K, repeat = False)
        pts1 = pts1.T
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
        points_3d = points_3d[:, 0, :]
        visualdep = get_visual_depth (points_3d)
        cam = get_velodyne_data(binary, calib)
        lidepth = cam[2, :]
        scale = lidepth / visualdep
        
    

image = superimposed_lidar_image(name, binary, calib)

# Create an Open3D PointCloud object for Velodyne point cloud
velodyne_point_cloud = o3d.geometry.PointCloud()
#velodyne_point_cloud.points = o3d.utility.Vector3dVector(np.transpose(cam))
velodyne_point_cloud.points = o3d.utility.Vector3dVector(cam.T[:, :3])



#Create an Open3D 3D points of images
points3d = o3d.geometry.PointCloud()
points3d.points = o3d.utility.Vector3dVector(points_3d)
# Set a uniform blue color for all points in the PLY file
blue_color = [0, 0, 1]  # RGB values for blue
points3d.paint_uniform_color(blue_color)


# Customize visualization settings
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(velodyne_point_cloud)
render_options = visualizer.get_render_option()
render_options.point_size = 5  # Adjust point size
visualizer.run()

vis_image = o3d.visualization.Visualizer()
vis_image.create_window()
vis_image.add_geometry(points3d)
vis_image.get_render_option().point_size = 5  # Adjust the point size here
vis_image.run()



visualizer.destroy_window()