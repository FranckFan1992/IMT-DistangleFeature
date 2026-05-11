import numpy as np
import collections
import struct
import sys
import os
from collections import defaultdict
from itertools import combinations

from .utils import *


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}

CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points_dict = {}

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            id_3d = np.array(binary_point_line_properties[0]) 
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            ids_img = list(map(float, track_elems[0::2]))
            ids_2dpts = list(map(float, track_elems[1::2]))
            points_dict[p_id] = {
                "id_3d": id_3d,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "image_ids": ids_img,
                "point2d_ids": ids_2dpts
            }
            
    return points_dict


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def readColmapCameras(cam_extrinsics, cam_intrinsics, test_images_name):

    max_id = max(cam_extrinsics.keys())
    image_name_list = [""] * (max_id+1)  # ex : img_id=768  image_name = 767.color.png 
    depth_name_list = [""] * (max_id+1)
    intrinsics_list = [np.zeros((3,3)) for _  in range(max_id+1)]
    pose_list = [np.zeros((4,4)) for _  in range(max_id+1)]
    
    train_idx_list = []
    test_idx_list = []

    for idx, key in enumerate(cam_extrinsics):  # key 的最小值是1 extr.id 最小值也是1  当extr.id =1 时 img name =  seq-01/frame-000000.color.png
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        img_name = extr.name
        
        # If the img_name is in the test_images_name, put it into a test id list, other wise in a train id list
        if img_name in test_images_name:
            test_idx_list.append(extr.id)
        else:
            train_idx_list.append(extr.id)
        
        image_name_list[key] = img_name
        depth_name_list[key] = img_name.replace(".color.png", ".depth.png")
        
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        ### elif intr.model=="PINHOLE":
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        

        intrinsic = getIntrinsic(FovX, height, width)
        pose = getExtrinsic(R,T)
        intrinsics_list[key] = intrinsic
        pose_list[key] = pose
        
    sys.stdout.write('\n')
    res = {
            "image_name_list": image_name_list,  # image_name_list[0]= ""
            "depth_name_list": depth_name_list,
            "intrinsics_list": intrinsics_list,
            "pose_list": pose_list,
            "train_idx_list": train_idx_list,
            "test_idx_list": test_idx_list,
    }
    
    return res


def loadSFM(data_path):
    
    sfm_images_path = os.path.join(data_path, "sparse/0", "images.bin")
    sfm_point3d_path = os.path.join(data_path, "sparse/0/", "points3D.bin")
    sfm_cameras_path = os.path.join(data_path, "sparse/0/", "cameras.bin")
    
    points = read_points3D_binary(sfm_point3d_path)
    images = read_extrinsics_binary(sfm_images_path)
    cameras = read_intrinsics_binary(sfm_cameras_path)
    
    if os.path.exists(os.path.join(data_path, "sparse/0", "list_test.txt")):
        
        # 7scenes
        with open(os.path.join(data_path, "sparse/0", "list_test.txt")) as f:
            test_images = f.readlines()
            test_images = [x.strip() for x in test_images]
    else:
        test_images = []
    
    return points, images, cameras, test_images

############################################
# 3 构造图像对 + sparse matches
############################################

def build_images_pairs(points_dict, images,
                           train_ids, test_ids,
                           frame_index,
                           min_frame_dist=10,
                           min_matches=30):

    train_pairs = defaultdict(list)
    test_pairs = defaultdict(list)

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    for p in points_dict.values():
        img_ids = np.array(p["image_ids"], dtype=int)
        pt2d_ids = np.array(p["point2d_ids"], dtype=int)
        n = len(img_ids)
        if n < 2:
            continue

        idx_i, idx_j = np.triu_indices(n, k=1)
        imgs_i = img_ids[idx_i]
        imgs_j = img_ids[idx_j]
        pts_i = pt2d_ids[idx_i]
        pts_j = pt2d_ids[idx_j]

        for k in range(len(idx_i)):
            img_i = imgs_i[k]
            img_j = imgs_j[k]
            if img_i not in images or img_j not in images:
                continue

            info_i = frame_index[img_i]
            info_j = frame_index[img_j]

            if info_i["seq"] == info_j["seq"] and abs(info_i["frame"] - info_j["frame"]) < min_frame_dist:
                continue

            xy_i = images[img_i].xys[pts_i[k]]
            xy_j = images[img_j].xys[pts_j[k]]

            if img_i < img_j:
                key = (img_i, img_j)
                match = [xy_i[0], xy_i[1], xy_j[0], xy_j[1]]
            else:
                key = (img_j, img_i)
                match = [xy_j[0], xy_j[1], xy_i[0], xy_i[1]]

            if img_i in train_ids and img_j in train_ids:
                train_pairs[key].append(match)
            elif ((img_i in train_ids and img_j in test_ids) or
                  (img_j in train_ids and img_i in test_ids)):
                test_pairs[key].append(match)
            # test-test 忽略

    def finalize_pairs(pairs_dict):
        new_dict = {}
        for k, v in pairs_dict.items():
            arr = np.array(v, dtype=np.float32)
            if len(arr) >= min_matches:
                new_dict[k] = arr
        return new_dict

    return finalize_pairs(train_pairs), finalize_pairs(test_pairs)



def build_multiview_groups(points_dict, images,
                           train_ids, test_ids,
                           frame_index,
                           min_frame_dist=20):
    """
    构建 multi-view training groups.
    
    返回：
        train_groups: list of dict, 每个 dict 包含:
            'image_ids': [V] 图像ID列表（整数）
            'coords': [V,2] 对应的2D点
        test_groups: 类似
    """
    train_ids = set(train_ids)
    test_ids = set(test_ids)

    train_groups = []
    test_groups = []

    for p in points_dict.values():

        img_ids = np.array(p["image_ids"], dtype=int)
        pt2d_ids = np.array(p["point2d_ids"], dtype=int)

        n = len(img_ids)
        if n < 2:
            continue

        valid_imgs = []
        valid_coords = []

        for i in range(n):

            img_i = int(img_ids[i])
            if img_i not in images:
                continue

            coord_i = images[img_i].xys[pt2d_ids[i]]

            valid = True

            # 检查与当前 group 所有图像的 frame distance
            for j, img_j in enumerate(valid_imgs):

                info_i = frame_index[img_i]
                info_j = frame_index[img_j]

                if info_i["seq"] == info_j["seq"] and abs(info_i["frame"] - info_j["frame"]) < min_frame_dist:
                    valid = False
                    break

            if valid:
                valid_imgs.append(img_i)
                valid_coords.append(coord_i)

        if len(valid_imgs) < 2:
            continue

        if all(img in train_ids for img in valid_imgs):

            train_groups.append({
                "image_ids": np.array(valid_imgs, dtype=np.int64),
                "coords": np.array(valid_coords, dtype=np.float32)
            })

    # test groups
    for img_id in test_ids:

        if img_id in images:

            test_groups.append({
                "image_ids": np.array([img_id], dtype=np.int64),
                "coords": np.array(images[img_id].xys, dtype=np.float32)
            })

    print("len(train_dataset) =", len(train_groups))

    return train_groups, test_groups