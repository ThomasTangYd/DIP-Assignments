import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import math

# 确保输出目录存在
os.makedirs('output', exist_ok=True)

# 数据加载
def load_data(num_views):
    points2d = np.load('data/points2d.npz')
    points3d_colors = np.load('data/points3d_colors.npy')

    return points2d, points3d_colors

# 初始化参数
def initialize_parameters(num_points, num_views):
    # 相机中心点
    image_width = 1024
    image_height = 1024
    cx = image_width / 2
    cy = image_height / 2
    
    # 焦距初始化
    # FoV设为60度，使用公式 f = H / (2 * tan(fov/2))
    fov = 60  # 角度
    fov_rad = math.radians(fov)
    f = image_height / (2 * math.tan(fov_rad / 2))
    f = torch.tensor([f], requires_grad=True, dtype=torch.float32)
    
    # 相机外参初始化
    # 旋转初始化为单位矩阵（Euler角为零）
    euler_angles = torch.zeros(num_views, 3, requires_grad=True, dtype=torch.float32)
    
    # 平移初始化为 [0, 0, -d]，d为合理的观测距离
    d = 2.5
    translations = torch.zeros(num_views, 3, dtype=torch.float32)
    translations[:, 2] = -d
    translations = translations.requires_grad_(True)
    
    # 3D点坐标初始化：在原点附近的随机位置
    points3d = torch.randn(num_points, 3, dtype=torch.float32)
    points3d = points3d * 0.1
    points3d.requires_grad_(True)
    
    return f, euler_angles, translations, points3d, cx, cy

# Euler角到旋转矩阵的转换 (XYZ约定)
def euler_angles_to_matrix(euler_angles):
    n_views = euler_angles.shape[0]
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw = euler_angles[:, 2]
    
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    # 创建批量旋转矩阵
    R = torch.zeros(n_views, 3, 3, dtype=euler_angles.dtype)
    
    # 计算旋转矩阵的每个元素
    R[:, 0, 0] = cos_yaw * cos_pitch
    R[:, 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    R[:, 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    R[:, 1, 0] = sin_yaw * cos_pitch
    R[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    R[:, 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    R[:, 2, 0] = -sin_pitch
    R[:, 2, 1] = cos_pitch * sin_roll
    R[:, 2, 2] = cos_pitch * cos_roll
    
    return R

# 投影函数
def project_points(points3d, euler_angles, translations, f, cx, cy):
    # 将Euler角转换为旋转矩阵
    R = euler_angles_to_matrix(euler_angles)
    
    # 3D点变换到相机坐标系
    points3d_expanded = points3d.unsqueeze(0)
    R_expanded = R.unsqueeze(1)
    translations_expanded = translations.unsqueeze(1)
    
    # 相机坐标系中的点
    # [Xc, Yc, Zc] = R @ P + T
    points_camera = torch.matmul(R_expanded, points3d_expanded.unsqueeze(-1)).squeeze(-1) + translations_expanded
    
    # 投影到图像平面
    Xc = points_camera[..., 0]
    Yc = points_camera[..., 1]
    Zc = points_camera[..., 2]
    
    # 应用投影公式
    u = -f * Xc / Zc + cx
    v = f * Yc / Zc + cy
    
    return u, v

# 计算损失函数
def compute_loss(u_pred, v_pred, points2d):
    loss = 0.0
    n_visible = 0
    
    # 处理所有视角
    for i, key in enumerate(points2d.keys()):
        obs = points2d[key]
        x_gt = torch.tensor(obs[:, 0], dtype=torch.float32)
        y_gt = torch.tensor(obs[:, 1], dtype=torch.float32)
        visibility = torch.tensor(obs[:, 2], dtype=torch.bool)
        
        # 只计算可见点的损失
        u_pred_view = u_pred[i, visibility]
        v_pred_view = v_pred[i, visibility]
        x_gt_view = x_gt[visibility]
        y_gt_view = y_gt[visibility]
        
        if len(u_pred_view) > 0:
            # 计算重投影误差
            error = torch.sqrt((u_pred_view - x_gt_view)**2 + (v_pred_view - y_gt_view)**2)
            loss += error.sum()
            n_visible += len(u_pred_view)
    
    # 平均损失
    if n_visible > 0:
        loss /= n_visible
    
    return loss

# 保存带颜色的3D点云为OBJ文件
def save_point_cloud(points3d, colors, filename):
    with open(filename, 'w') as f:
        for i in range(len(points3d)):
            x, y, z = points3d[i].detach().numpy()
            r, g, b = colors[i]
            f.write(f"v {x} {y} {z} {r} {g} {b}\n")
    print(f"已保存重建的3D点云：{filename}")

# 保存相机参数到文件
def save_camera_parameters(f, euler_angles, translations, filename):
    with open(filename, 'w', encoding='utf-8') as f_out:
        # 写入焦距
        f_out.write(f"# 焦距\n")
        f_out.write(f"focal_length: {f.item():.4f}\n\n")
        
        # 写入相机外参
        f_out.write("# 相机外参\n")
        f_out.write("# 格式: view_id roll pitch yaw tx ty tz\n")
        
        num_views = euler_angles.shape[0]
        for i in range(num_views):
            roll, pitch, yaw = euler_angles[i].detach().numpy()
            tx, ty, tz = translations[i].detach().numpy()
            f_out.write(f"{i} {roll:.6f} {pitch:.6f} {yaw:.6f} {tx:.6f} {ty:.6f} {tz:.6f}\n")
    print(f"已保存相机参数：{filename}")

# 主函数
def main():
    # try:
    # 设置使用的视角数量
    num_views = 50
    
    # 加载数据
    points2d, points3d_colors = load_data(num_views)
    
    # 获取点的数量
    num_points = len(points3d_colors)
    
    # 初始化参数
    f, euler_angles, translations, points3d, cx, cy = initialize_parameters(num_points, num_views)
    
    # 优化器
    optimizer = torch.optim.Adam([f, euler_angles, translations, points3d], lr=0.01)
    
    # 优化
    n_epochs = 300
    losses = []

    for epoch in range(n_epochs):
        # 前向传播：投影点
        u_pred, v_pred = project_points(points3d, euler_angles, translations, f, cx, cy)
        
        # 计算损失
        loss = compute_loss(u_pred, v_pred, points2d)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        losses.append(loss.item())
        
        # 打印进度
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    # 可视化损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_epochs), losses)
    plt.title('Bundle Adjustment Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reprojection Error')
    plt.grid(True)
    plt.savefig('output/loss_curve.png')
    print("已保存损失曲线：output/loss_curve.png")
    
    # 保存重建的3D点云为OBJ文件
    save_point_cloud(points3d, points3d_colors, 'output/reconstructed_point_cloud.obj')
    
    # 保存相机参数到文件
    save_camera_parameters(f, euler_angles, translations, 'output/camera_parameters.txt')

if __name__ == "__main__":
    main()