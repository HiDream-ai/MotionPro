import torch
import numpy as np
from torchvision.transforms import ToTensor
import os
import cv2


class CameraParams:             # coordinary system---https://zhuanlan.zhihu.com/p/593204605/
    def __init__(self, H: int = 512, W: int = 512):
        self.H = H
        self.W = W
        self.focal = (5.8269e+02, 5.8269e+02)
        self.fov = (2*np.arctan(self.W / (2*self.focal[0])), 2*np.arctan(self.H / (2*self.focal[1])))
        self.K = np.array([
            [self.focal[0], 0., self.W/2],
            [0., self.focal[1], self.H/2],
            [0.,            0.,       1.],
        ]).astype(np.float32)

def tilt_pan(deg, num_frames, mode):
    degsum = deg
    # 512x512
    pre_angle = 60
    pre_num = 4
    if mode in ['left','right']:
        if mode == 'left':
            pre = np.linspace(0, -pre_angle, pre_num)
            thlist = np.concatenate((pre, np.linspace(0, -degsum, num_frames)))
        elif mode == 'right':
            pre = np.linspace(0, pre_angle, pre_num)
            thlist = np.concatenate((pre, np.linspace(0, degsum, num_frames)))
        philist = np.zeros_like(thlist)
    elif mode in ['up','down']:
        if mode == 'up':
            pre = np.linspace(0, -pre_angle, pre_num)
            philist = np.concatenate((pre, np.linspace(0, -degsum, num_frames)))
        elif mode == 'down':
            pre = np.linspace(0, pre_angle, pre_num)
            philist = np.concatenate((pre, np.linspace(0, degsum, num_frames)))
        thlist = np.zeros_like(philist)
    assert len(thlist) == len(philist)
    zero_index = len(pre)
    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))
    return render_poses, zero_index

def roll(deg, num_frames, mode):
    degsum = deg
    if mode == 'anticlockwise':
        galist = np.linspace(0, degsum, num_frames)
    elif mode == 'clockwise':
        galist = np.linspace(0, -degsum, num_frames)
    render_poses = np.zeros((len(galist), 3, 4))
    for i in range(len(galist)):
        ga = galist[i]
        render_poses[i,:3,:3] = np.array([[np.cos(ga/180*np.pi), -np.sin(ga/180*np.pi), 0], [np.sin(ga/180*np.pi), np.cos(ga/180*np.pi), 0], [0, 0, 1]])
        render_poses[i,:3,3:4] = np.zeros((3,1))
    zero_index = 0
    return render_poses, zero_index


def pedestal_truck(dis, num_frames, mode):
    pre_dis = 2
    pre_num = 5
    if mode in ['right','down']:
        pre = np.linspace(0, -pre_dis, pre_num)
        movement = np.concatenate((pre, np.linspace(0, -dis, num_frames)))
    elif mode in ['left','up']:
        pre = np.linspace(0, pre_dis, pre_num)
        movement = np.concatenate((pre, np.linspace(0, dis, num_frames)))
    render_poses = np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        render_poses[i,:3,:3] = np.eye(3)
        if mode in ['right','left']:
            render_poses[i,:3,3:4] = np.array([[movement[i]], [0], [0]])
        elif mode in ['up','down']:
            render_poses[i,:3,3:4] = np.array([[0], [movement[i]], [0]])
    zero_index = len(pre)
    return render_poses, zero_index

def zoom(dis, num_frames, mode):
    if mode == 'out':
        pre_dis =  1
        pre_num = 2
    elif mode == 'in':
        pre_dis = 1
        pre_num = 2
    if mode == 'out':
        pre = np.linspace(0, pre_dis, pre_num)              # NOTE: why add pre_dis, generate more cloud points to fix too large pose change
        movement = np.concatenate((pre, np.linspace(0, dis, num_frames)))
    elif mode == 'in':
        pre = np.linspace(0, -pre_dis, pre_num)
        movement = np.concatenate((pre, np.linspace(0, -dis, num_frames)))
    render_poses = np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        render_poses[i,:3,:3] = np.eye(3)                               # no rotation..
        render_poses[i,:3,3:4] = np.array([[0], [0], [movement[i]]])    # only translation..
    zero_index = len(pre)
    return render_poses, zero_index

def hybrid_out_left_up_down(dis, num_frames, mode):
    zoom_mode = 'out'
    zoom_dis = 2
    zoom_num_frames = num_frames
    zoom_pre_dis = 1
    zoom_pre_num = 2
    move_dis = 30
    move_num_frames = num_frames
    move_pre_dis = 30
    move_pre_num = 2
    if zoom_mode == 'out':
        zoom_pre = np.linspace(0, zoom_pre_dis, zoom_pre_num)
        zoom_movement = np.concatenate((zoom_pre, np.linspace(0, zoom_dis, zoom_num_frames)))
    elif zoom_mode == 'in':
        zoom_pre = np.linspace(0, -zoom_pre_dis, zoom_pre_num)
        zoom_movement = np.concatenate((zoom_pre, np.linspace(0, -zoom_dis, zoom_num_frames)))

    move_pre = np.linspace(0, move_pre_dis, move_pre_num)
    thlist = np.concatenate((move_pre, np.linspace(0, move_dis, move_num_frames)))

    move_pre = np.linspace(0, move_pre_dis, move_pre_num)
    philist = np.concatenate((move_pre, np.linspace(0, move_dis, move_num_frames)))
    render_poses = np.zeros((len(philist), 3, 4))


    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.array([[zoom_movement[i]], [zoom_movement[i]], [zoom_movement[i]]])
    assert len(zoom_pre) == len(move_pre)
    zero_index = len(zoom_pre)
    return render_poses, zero_index

def hybrid_in_then_up(dis, num_frames, mode):
    zoom_dis = 2
    zoom_num_frames = num_frames
    zoom_pre_dis = 1
    zoom_pre_num = 2
    zoom_pre = np.linspace(0, -zoom_pre_dis, zoom_pre_num)
    zoom_movement = np.concatenate((zoom_pre, np.linspace(0, -zoom_dis, zoom_num_frames)))
    render_poses = np.zeros((len(zoom_movement), 3, 4))

    for i in range(len(zoom_movement)):
        render_poses[i,:3,:3] = np.eye(3)
        if i < len(zoom_pre):
            render_poses[i,:3,3:4] = np.array([[0], [-zoom_movement[i]], [0]])
        else:
            mem = (len(zoom_pre) + zoom_num_frames//2)
            if i <  mem:
                render_poses[i,:3,3:4] = np.array([[0], [0], [zoom_movement[i]]])
            else:
                fix=zoom_movement[mem-1]
                render_poses[i,:3,3:4] = np.array([[0], [-zoom_movement[i-mem+2]], [fix]])
    zero_index = len(zoom_pre)
    return render_poses, zero_index

def rotate(deg, num_frames, mode, center_depth):
    degsum = deg
    if mode == 'clockwise':
        thlist = np.linspace(0, degsum, num_frames)
    elif mode == 'anticlockwise':
        thlist = np.linspace(0, -degsum, num_frames)
    phi = 0
    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        # d = 4.3 # manual central point for arc / you can change this value
        d = center_depth
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
    return render_poses, 0


def read_traj_from_RealEstate10k(pose_file, ratio):
    
    h,w = ratio
    with open(pose_file, 'r') as f:
        poses = f.readlines()
    
    w2cs = [np.asarray([float(p) for p in pose.strip().split(' ')[7:]]).reshape(3, 4) for pose in poses[1:]]
    fxs = [[float(pose.strip().split(' ')[1]), float(pose.strip().split(' ')[2])] for pose in poses[1:]]
    
    assert fxs[0] == fxs[1], 'impossible...'
    cam_intrinsic =  np.array([
                        [fxs[0][0]*w, 0.,      0.5*w],
                        [0.,    fxs[0][1]*h,   0.5*h],
                        [0.,            0.,       1.],
                    ]).astype(np.float32)
    
    traj_pose = np.stack(w2cs, axis=0)

    return traj_pose, cam_intrinsic

class Warper:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.cam = CameraParams(self.H, self.W)
        
        # depth model
        self.d_model = torch.hub.load('tools/ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')

    def d(self, im):
        return self.d_model.infer_pil(im)

    def get_traj_from_camera_pose(self, rgb_cond, camera_motion, num_frames=16):
        '''
        args:
            rgb_cond: First frame PIL.Image formation
            camera_motion: Define configs to create target camera_poses
            num_frames: Video length
        
        return:
            input_drag_dense: Optical flow seq ,which represent camera moving -- shape [f, H, W, 2], channls [Vx, Vy]
            visible_mask: Specify which points on the first frame are visible -- shape [H, W, 1]
        '''
        image = ToTensor()(rgb_cond)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).to('cuda')

        # Stage_1 gets camera pose and inver-project points into 3D space
        w_in, h_in = rgb_cond.size
        image_curr = rgb_cond
        depth_curr = self.d(image_curr)
        center_depth = np.mean(depth_curr[h_in//2-10:h_in//2+10, w_in//2-10:w_in//2+10])
        render_poses, _, newk = self.get_pcdGenPoses(camera_motion, center_depth, num_frames, K=self.cam.K, ratio=(h_in, w_in))       # (F, 3, 4) camera_poses
        if newk is not None:
            self.cam.K = newk
        
        H, W, K = self.cam.H, self.cam.W, self.cam.K
        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels

        R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]
        pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1)) # shape: [3(ux,uy,1), H,W]
        new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2--- shape: [3(X,Y,Z), H,W]
        new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)

        pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()

        # Stage_2 computes per_pose trajectory_drag
        input_drag_dense = torch.zeros(num_frames, H, W, 2)
        points_2d_old = np.stack((x, y), axis=0).transpose(1,2,0).reshape(-1,2)
        vis_mask = np.ones((H,W), dtype=np.float32)
                  
        for i in range(1, len(render_poses)):
            
            current_drag = np.zeros((H,W,2)).reshape(-1,2)
            R, T = render_poses[i,:3,:3], render_poses[i,:3,3:4]

            pts_coord_cam2 = R.dot(pts_coord_world) + T                     # rotation first..
            pixel_coord_cam2 = np.matmul(K, pts_coord_cam2)

            valid_idx = np.where(np.logical_and.reduce((pixel_coord_cam2[2]>0, 
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]>=0, 
                                                        pixel_coord_cam2[0]/pixel_coord_cam2[2]<=W-1, 
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]>=0, 
                                                        pixel_coord_cam2[1]/pixel_coord_cam2[2]<=H-1)))[0]
            pixel_coord_cam2 = pixel_coord_cam2[:2, valid_idx]/pixel_coord_cam2[-1:, valid_idx]
            # round_coord_cam2 = np.round(pixel_coord_cam2).astype(np.int32)

            current_drag[valid_idx] = pixel_coord_cam2.transpose(1,0) - points_2d_old[valid_idx]
            input_drag_dense[i] = torch.from_numpy(current_drag.reshape(H,W,2))

            visible_2d_points_old = points_2d_old[valid_idx].transpose(1,0).astype(np.int32)
            first_frame_vis_mask = np.zeros((H,W), dtype=np.float32)            
            first_frame_vis_mask[visible_2d_points_old[1], visible_2d_points_old[0]] = 1
            vis_mask = np.logical_and(vis_mask, first_frame_vis_mask).astype(np.float32)
        
        
        return input_drag_dense, vis_mask
    

    def get_pcdGenPoses(self, camera_motion, center_depth, num_frames=16, K=None, ratio=(320,512)):
        motion_class, mode, deg = camera_motion.split('-')
        if deg == 'default': deg = 1
        else:  deg = int(deg)
        pose_save_path = os.path.join('all_results/test/motionpro_s_pure_camera/pose_files', f'{camera_motion}.txt')
        
        zero_index = 0
        new_k = None
        if motion_class == 'zoom':
            render_poses, zero_index = zoom(deg,num_frames,mode)                         # w2c camera pose
        elif motion_class in ['tilt','pan']:
            render_poses, zero_index = tilt_pan(deg,num_frames,mode)
        elif motion_class in ['pedestal','truck']:
            render_poses, zero_index = pedestal_truck(deg,num_frames,mode)
        elif motion_class == 'roll':
            render_poses, zero_index = roll(deg,num_frames,mode)
        elif motion_class == 'rotate':
            render_poses, zero_index = rotate(deg,num_frames,mode, center_depth)
        elif motion_class == 'hybrid':
            if mode == 'in_then_up':
                render_poses, zero_index = hybrid_in_then_up(deg,num_frames,mode)
            elif mode == 'out_left_up_down':
                render_poses, zero_index = hybrid_out_left_up_down(deg,num_frames,mode)
        elif motion_class == 'complex':
            render_poses = np.zeros((num_frames, 3, 4))
            if mode == 'mode_1':
                trajectories = torch.load('assets/pose_files/complex_1.pth').reshape([16, 3, 4])
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_2':
                trajectories = torch.load('assets/pose_files/complex_2.pth').reshape([16, 3, 4])
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_3':
                trajectories = torch.load('assets/pose_files/complex_3.pth').reshape([16, 3, 4])
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_4':
                trajectories = torch.load('assets/pose_files/complex_4.pth').reshape([16, 3, 4])
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_5':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/0bf152ef84195293.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_6':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/0c9b371cc6225682.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_7':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/0c11dbe781b1c11c.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_8':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/0f47577ab3441480.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_9':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/0f68374b76390082.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_10':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/2c80f9eb0d3b2bb4.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_11':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/2f25826f0d0ef09a.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_12':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/3c35b868a8ec3433.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_13':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/3f79dc32d575bcdc.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            elif mode == 'mode_14':
                trajectories, new_k = read_traj_from_RealEstate10k('assets/pose_files/4a2d6753676df096.txt', ratio)
                render_poses[zero_index:] = trajectories[:num_frames]
            else:
                assert False, 'Impossible..'
        else:
            raise("Invalid pcdgenpath")

        # diff from ori code..
        render_poses = render_poses[zero_index:]
        if new_k is not None:
            K = new_k
        
        k_np = np.array([K[0][0]/ratio[1], K[1][1]/ratio[0], 0.5, 0.5, 0.0, 0.0])             # TODO: FOR visualization
        os.makedirs(os.path.dirname(pose_save_path), exist_ok=True)
        with open(pose_save_path, 'w') as f:
            for i in range(num_frames):
                if i==0: f.write(f'{camera_motion}'+"\n")
                line_txt = f'{i+1:03d} ' + ' '.join(map(str, k_np.flatten())) + ' ' + ' '.join(map(str, render_poses[i].flatten()))
                f.write(line_txt + "\n")
        
        
        return render_poses, zero_index, new_k

    
    def process_into_sparse(self, input_drag_dense, visible_mask):
        '''
        args:
            input_drag_dense: Estimated optical flow seq, shape [f,h,w,2]
            visible_mask: Visible points in the first frame
            
        return:
            input_drag: Sparsified visible trajectories
            motion_bucket_id: Compute mean flow_mag -- first compute per_frame mean flow_mag    
        '''
        
        self.block_size = 8
        f,h,w,c = input_drag_dense.shape
        
        # Compute motion_bucket_id
        all_means_list = []
        for i in range(f):
            flow_magnitude = torch.sum(input_drag_dense[i]**2, dim=-1).sqrt()  # [T, H, W]
            nonzero_mask = flow_magnitude > 0  # [T, H, W]
            if nonzero_mask.any():
                mean_flow_magnitude = torch.mean(flow_magnitude[nonzero_mask])
            else:
                mean_flow_magnitude = torch.tensor(0.0) 
            
            all_means_list.append(mean_flow_magnitude)
        
        motion_bucket_id = torch.mean(torch.tensor(all_means_list)).to(dtype=input_drag_dense.dtype)
        
        # Get sparse input_drag
        mask_ratio = 0.85
        block_mask = np.random.rand(h//self.block_size, w//self.block_size) > mask_ratio
        mask_numpy_final_resized = cv2.resize(visible_mask, (w//self.block_size, h//self.block_size), interpolation=cv2.INTER_NEAREST) 
        block_mask_vis = np.logical_and(block_mask, mask_numpy_final_resized).astype(np.float32)
        
        full_mask = np.kron(block_mask_vis, np.ones((self.block_size, self.block_size), dtype=np.uint8))
        full_mask = torch.from_numpy(full_mask).to(torch.float32)[None,:,:,None].repeat(f,1,1,1)
        input_drag = input_drag_dense * full_mask
        
        motion_bucket_id = torch.ones_like(motion_bucket_id) * 17
        
        return input_drag, motion_bucket_id


