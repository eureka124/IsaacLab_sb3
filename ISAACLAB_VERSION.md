# Isaac Lab 版本记录

记录日期：2026-09-02

## 升级前

- 仓库路径：`/home/hxy/IsaacLab`
- 分支：`main`
- Isaac Lab 提交：`64a97f205fd14830f5df9759ce3e44b7515ef8f5`
- `git describe`：`v2.2.1-109-g64a97f205fd`
- `isaaclab` 扩展版本：`0.47.7`
- Isaac Sim 运行时版本：`5.0.0-rc.45+release.23960.184afb15.gl`
- Isaac Sim 安装目录：`/home/hxy/isaacsim`
- 本地回退分支：`backup/isaaclab-pre-upgrade-20260902`

## 升级目标

- Isaac Lab 发布版：`v2.3.2`
- 提交：`37ddf626871758333d6ed89cf64ad702aef127d0`
- `isaaclab` 扩展版本：`0.54.2`
- 选择原因：这是支持 Isaac Sim 5.1 的最新 2.x 稳定版，并包含 Multi-Mesh RayCaster；Isaac Lab 3.0 Beta 要求 Isaac Sim 6.0。

## 升级结果

- 当前分支：`local/v2.3.2`
- 当前提交：`37ddf626871758333d6ed89cf64ad702aef127d0`
- Isaac Sim 版本：`5.1.0-rc.19+release.26219.9c81211b.gl`
- Isaac Sim 安装目录：`/home/hxy/isaacsim-5.1.0`
- `/home/hxy/IsaacLab/_isaac_sim` 已指向新安装目录；旧版 `/home/hxy/isaacsim` 保留。
- 官方 5.1.0 独立包内置 URDF importer `2.4.30`。由于本项目只使用预转换 USD，
  `apps/isaaclab.python.kit` 已取消对 registry-only `2.4.31` 的精确约束，使用本地可用版本。
- Conda 与 Isaac Sim Python 的 editable 包均已更新。
- Conda 激活脚本可从新 Isaac Sim 目录加载构建依赖，`isaaclab` editable 源码路径保持为
  `/home/hxy/IsaacLab/source/isaaclab`。
- `Training-Mazes-v0` 已通过 1 环境的 headless 初始化、GUI 初始化和 1 个 PPO 迭代测试。
- GUI 测试成功创建环境窗口，RTX 5080、Vulkan、PhysX 和 CUDA 均正常识别。
- 1 迭代测试日志位于
  `/home/hxy/tutorial/logs/2026-09-02_15-28-37/isaacsim-5.1-upgrade-smoke`。
- Isaac Sim 5.1.0 内置 `omni.warp.core` 为 1.8.2，而 v2.3.2 使用更新的 Warp API；`scripts/sb3/train.py` 会在 `AppLauncher` 前预加载 Conda 环境中的 Warp 1.10.1。

## 深度相机选择

- 训练迷宫使用 `MultiMeshRayCasterCamera`。
- 保持原来的 48×64 针孔参数和 `distance_to_image_plane` 输出，策略观测仍为 `(1, 12, 16)`。
- 墙体与圆柱启用网格变换跟踪，以反映 reset 时的迷宫布局和随机障碍位置。
- 训练不再依赖 RTX 相机渲染，因此 headless 训练无需 `--enable_cameras`。

## 回退方法

先把 Isaac Sim 链接恢复到旧版：

```bash
ln -sfnT /home/hxy/isaacsim /home/hxy/IsaacLab/_isaac_sim
```

如果还需要回退 Isaac Lab，再切换升级前分支：

```bash
git -C /home/hxy/IsaacLab switch backup/isaaclab-pre-upgrade-20260902
```

切换源码后，如果需要让 Python 包的版本元数据也恢复到旧版，在 Conda 环境与 Isaac Sim Python 中分别重新安装 editable 包：

```bash
cd /home/hxy/IsaacLab
/home/hxy/miniconda3/envs/env_isaaclab/bin/python -m pip install --no-deps \
  -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_mimic \
  -e source/isaaclab_rl -e source/isaaclab_tasks
/home/hxy/isaacsim/python.sh -m pip install --no-deps \
  -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_mimic \
  -e source/isaaclab_rl -e source/isaaclab_tasks
```

也可以直接检出升级前提交：

```bash
git -C /home/hxy/IsaacLab switch --detach 64a97f205fd14830f5df9759ce3e44b7515ef8f5
```
