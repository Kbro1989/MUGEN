
import os
import torch
import time
import imageio
import numpy as np
import moviepy.editor as mp
from scipy.spatial.transform import Rotation as RRR
import motGPT.render.matplot.plot_3d_global as plot_3d
from motGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from motGPT.render.pyrender.smpl_render import SMPLRender

SMPL_MODEL_PATH = 'deps/smpl_models/smpl'

def render_motion(data, feats, output_dir, fname=None, method='fast', smpl_model_path=SMPL_MODEL_PATH, fps=20):
    if fname is None:
        fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
            time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    # np.save(output_npy_path, feats)

    if method == 'slow':
        if len(data.shape) == 4:
            data = data[0]
        data = data - data[0, 0]
        pose_generator = HybrIKJointsToRotmat()
        pose = pose_generator(data)
        pose = np.concatenate([
            pose,
            np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
        ], 1)
        shape = [768, 768]
        render = SMPLRender(smpl_model_path)

        r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
        vid = []
        aroot = data[:, 0].copy()
        aroot[:, 1] = -aroot[:, 1]
        aroot[:, 2] = -aroot[:, 2]
        params = dict(pred_shape=np.zeros([1, 10]),
                      pred_root=aroot,
                      pred_pose=pose)
        render.init_renderer([shape[0], shape[1], 3], params)
        for i in range(data.shape[0]):
            renderImg = render.render(i)
            vid.append(renderImg)

        # out = np.stack(vid, axis=0)
        out_video = mp.ImageSequenceClip(vid, fps=fps)
        out_video.write_videofile(output_mp4_path, fps=fps)
        del render

    elif method == 'fast':
        output_gif_path = output_mp4_path[:-4] + '.gif'
        if len(data.shape) == 3:
            data = data[None]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        pose_vis = plot_3d.draw_to_batch(data, [''], None, fps=fps)[0].cpu().numpy()

        out_video = mp.ImageSequenceClip(list(pose_vis),fps=fps)
        out_video.write_videofile(output_mp4_path, fps=fps)
        # out_video = mp.VideoClip(make_frame=lambda t:pose_vis[int(t*fps)], duration=len(pose_vis)/fps)
        # out_video.write_videofile(output_mp4_path,fps=fps)
        del pose_vis


def _silence_worker_warnings():
    """ProcessPoolExecutor initializer: silence noisy third-party warnings that
    fire when each spawned worker imports torch / lightning_fabric / pydantic.

    Runs once per worker, before any task. Also sets PYTHONWARNINGS so that any
    further subprocess (e.g. moviepy's ffmpeg wrapper) inherits the same.
    """
    import warnings as _w
    import os as _os
    for cat in (FutureWarning, UserWarning, DeprecationWarning, ImportWarning):
        _w.filterwarnings("ignore", category=cat)
    _w.filterwarnings("ignore")
    _os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _render_clip_fast_numpy(joints_np, fps: int = 20, figsize=(5.0, 5.0),
                            dpi: int = 64, radius: float = 4.0):
    """Render a single motion clip to a (T, H, W, 3) uint8 numpy array.

    Much faster than `plot_3d_global.plot_3d_motion`: creates the figure/axes
    once per clip and only updates line data per frame. Also defaults to a
    smaller figsize/dpi suited for preview videos.
    """
    import io as _io
    import numpy as _np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _Poly3DCollection
    import mpl_toolkits.mplot3d.axes3d as _p3

    data = _np.asarray(joints_np, dtype=_np.float32).copy().reshape(
        len(joints_np), -1, 3)
    nb_joints = data.shape[1]
    # Pick the kinematic chain by joint count: HumanML3D SMPL-22 (default), its
    # 21-joint variant, or SnapMoGen's 24-joint skeleton (official
    # SnapMoGen/codes/animate.py kinematic_chain). All three share the same
    # 5-chain colour scheme below.
    if nb_joints == 21:
        chains = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20],
                  [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
    elif nb_joints == 24:
        chains = [[0, 1, 2, 3, 4, 5, 6], [3, 7, 8, 9, 10], [3, 11, 12, 13, 14],
                  [0, 15, 16, 17, 18, 19], [15, 20, 21, 22, 23]]
    else:
        chains = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                  [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                  [9, 13, 16, 18, 20]]
    colors = ['red', 'blue', 'black', 'red', 'blue']
    linewidths = [4.0, 4.0, 4.0, 4.0, 4.0]

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]].copy()
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    # SnapMoGen joint positions are recovered in the raw BVH scale (~cm, ~100x
    # HumanML3D's metres), so the fixed metres-scale view box would push the
    # figure off-screen. Auto-fit the cube to the root-centred body extent.
    if nb_joints == 24:
        _yt = float(data[..., 1].max())
        _xz = float(max(_np.abs(data[..., 0]).max(), _np.abs(data[..., 2]).max()))
        radius = max(2.0 * _xz, _yt, 1.0) * 1.1

    fig = _plt.figure(figsize=figsize, dpi=dpi)
    ax = _p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([0, radius])
    ax.view_init(elev=110, azim=-90)
    ax.set_axis_off()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    try:
        ax.dist = 7.5
    except Exception:
        pass

    # Pre-create line artists for chains + trajectory; update data per frame.
    chain_lines = []
    for chain, color, lw in zip(chains, colors, linewidths):
        (ln,) = ax.plot3D(
            data[0, chain, 0], data[0, chain, 1], data[0, chain, 2],
            linewidth=lw, color=color,
        )
        chain_lines.append((ln, chain))
    (traj_line,) = ax.plot3D([0.0], [0.0], [0.0], linewidth=1.0, color='blue')
    plane_collection = [None]

    def _draw_plane(index):
        if plane_collection[0] is not None:
            try:
                plane_collection[0].remove()
            except Exception:
                pass
        verts = [[
            [MINS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1]],
            [MINS[0] - trajec[index, 0], 0, MAXS[2] - trajec[index, 1]],
            [MAXS[0] - trajec[index, 0], 0, MAXS[2] - trajec[index, 1]],
            [MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1]],
        ]]
        poly = _Poly3DCollection(verts)
        poly.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(poly)
        plane_collection[0] = poly

    frames = []
    T = data.shape[0]
    for index in range(T):
        _draw_plane(index)
        if index > 1:
            traj_line.set_data_3d(
                trajec[:index, 0] - trajec[index, 0],
                _np.zeros(index, dtype=_np.float32),
                trajec[:index, 1] - trajec[index, 1],
            )
        else:
            traj_line.set_data_3d([0.0], [0.0], [0.0])
        for ln, chain in chain_lines:
            ln.set_data_3d(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
            )
        buf = _io.BytesIO()
        fig.savefig(buf, format='raw', dpi=dpi)
        buf.seek(0)
        w, h = int(fig.bbox.bounds[2]), int(fig.bbox.bounds[3])
        arr = _np.frombuffer(buf.getvalue(), dtype=_np.uint8).reshape(h, w, -1)
        # RGBA -> RGB (raw matplotlib buffer is RGBA).
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        frames.append(arr.copy())
        buf.close()

    _plt.close(fig)
    return _np.stack(frames, axis=0)


def render_fast_to_file(joints_np, output_mp4_path: str, fps: int = 20,
                        figsize=(5.0, 5.0), dpi: int = 64):
    """Subprocess-safe worker: render joints (numpy, shape (T,J,3) or (1,T,J,3)) to MP4.

    Uses a per-clip figure-reuse renderer for ~5-10x speedup vs
    `plot_3d_global.plot_3d_motion`. Writes via imageio's ffmpeg backend to
    avoid moviepy's per-call overhead.
    """
    # Silence noisy third-party warnings emitted at import time inside each
    # spawned worker (pynvml/FutureWarning from torch, pkg_resources from
    # lightning_fabric, etc.). Must happen before the imports below.
    import warnings as _warnings
    _warnings.filterwarnings("ignore", category=FutureWarning)
    _warnings.filterwarnings("ignore", category=UserWarning)
    _warnings.filterwarnings("ignore", category=DeprecationWarning)

    import numpy as _np
    import imageio as _imageio

    arr = _np.asarray(joints_np)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"render_fast_to_file: expected (T,J,3), got {arr.shape}")

    frames = _render_clip_fast_numpy(arr, fps=fps, figsize=figsize, dpi=dpi)
    # imageio ffmpeg writer; macro_block_size=1 lets odd resolutions through.
    _imageio.mimwrite(
        output_mp4_path, frames, fps=fps,
        codec='libx264', quality=6, macro_block_size=1,
    )
    del frames
    return output_mp4_path
