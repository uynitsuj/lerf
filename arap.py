from typing import Tuple, Optional
from copy import deepcopy

import tqdm

import torch
import torch.nn.utils

torch.set_printoptions(sci_mode=False)

from torch.nn import ParameterDict
from torchvision.transforms.functional import resize

from cuml.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
from threading import Lock

from gsplat._torch_impl import quat_to_rotmat

# For the nerfstudio config loading.
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer

# DIG imports
from garfield.garfield_gaussian_pipeline import GarfieldGaussianPipeline

import warnings
warnings.filterwarnings("ignore")

# Why am I seeing this "spitting out" behavior?
# (Maybe this isn't giving a cue on *how* to improve the distance?)
# (maybe you want more like... an L1 loss on it.) The max_dist isn't actually being decreased / affected.


def load_viewer_from_config(
    config: Path,
) -> Tuple[Viewer, GarfieldGaussianPipeline]:
    """Load garfield+dino pipeline and viewer from a config file."""
    train_config, pipeline, _, _ = eval_setup(config)

    # Must load a GarfieldGaussianPipelineConfig.
    assert type(pipeline) == GarfieldGaussianPipeline

    # We need to set up the writer to track number of rays,
    # otherwise the viewer will not calculate the resolution correctly.
    train_config.logging.local_writer.enable = False
    writer.setup_local_writer(
        train_config.logging, max_iter=train_config.max_num_iterations
    )

    return (
        Viewer(
            config=ViewerConfig(default_composite_depth=False, num_rays_per_chunk=-1),
            log_filename=config.parent,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            train_lock=Lock(),
        ),
        pipeline,
    )


def get_affinity_at_scale(
    pipeline: GarfieldGaussianPipeline, points: torch.Tensor, scale: float
) -> torch.Tensor:
    """Get the affinity of the gaussian means, at the provided Garfield scale."""
    grouping_feats = pipeline.garfield_pipeline[0].model.get_grouping_at_points(
        points, scale
    )
    return grouping_feats


def get_vid_frame(cap: cv2.VideoCapture, timestamp: float) -> np.ndarray:
    """Load a frame from a video at a specific timestamp, return RGB (in range [0...1])."""
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame number based on the timestamp and fps
    frame_number = min(int(timestamp * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))

    # Set the video position to the calculated frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap.read()

    # convert BGR to RGB
    frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    # Convert to [0...1] range.
    frame = frame / 255.0

    return frame


class ARAPOptimizer:
    pipeline: GarfieldGaussianPipeline
    gauss_params: ParameterDict
    """Pointer to self.pipeline.model.gauss_params."""
    init_c2o: Cameras
    """Initial estimate for camera position, with respect to the object."""
    num_gaussians: int
    """Number of gaussians in the pipeline/model."""
    global_pose_delta: torch.Tensor
    """Optimizable pose parameters, of object: [position (3), rotation (4)]."""
    local_pose_deltas: torch.Tensor
    """Optimizable pose parameters, for each gaussian *within* object: [position (3), rotation (4)]."""
    optimizer: torch.optim.Optimizer
    """Optimizer for the pose parameters."""
    init_means: torch.Tensor
    """Initial means of the gaussians."""
    init_quats: torch.Tensor
    """Initial quaternions of the gaussians."""
    nearest_ids: torch.Tensor
    """Indices of nearest neighbors for each gaussian."""
    render_lock: Lock
    """Lock for rendering, to prevent gaussian / viewer state bugs."""
    num_neighbors: int
    """Number of nearest neighbors to consider for each gaussian, for ARAP."""

    # make nieghbors matter more/less based on the distance
    # less neighbors probably good
    # miniize the transform, not the dist change.
    # would be cool if we could color the mesh with some rigidity.

    # What is happening to the occluded objects?
    # Normalizing... 
    # Initialize global pretty well before local.
    # Rigidity on the quaternions... (dynamic gaussian paper has something for this - jk.)
    # The diff in quaternion distance bweten it and its neighbors should be close to zero. 

    def __init__(
        self,
        pipeline: GarfieldGaussianPipeline,
        device: torch.device,
        render_lock: Lock,
        init_c2o: Optional[Cameras] = None,
        # lr: float = 0.001,
        lr: float = 0.01,
        num_neighbors: int = 10,
    ):
        self.pipeline = pipeline
        self.device = device

        # If no initial camera position is provided, use the first camera in the training dataset.
        if init_c2o is None:
            train_dataset = self.pipeline.datamanager.train_dataset
            assert (train_dataset is not None) and (len(train_dataset) > 0)
            init_c2o = train_dataset.cameras[:1]  # instead of [0], to keep shape[0] == 1.
            # make init_c2o's width == 500.
            init_c2o.rescale_output_resolution(
                500 / max(init_c2o.width.item(), init_c2o.height.item())
            )

        self.init_c2o = deepcopy(init_c2o).to(device)

        self.gauss_params = self.pipeline.model.gauss_params
        self.num_gaussians = self.gauss_params["means"].shape[0]

        # Cache the initial mean and quaternion of the gaussians, for reset, etc.
        self.init_means = self.gauss_params["means"].detach().clone()
        self.init_quats = self.gauss_params["quats"].detach().clone()

        # Optimizable parameters: [position (3), rotation (4)]. Optimize global and local pose deltas separately.
        # TODO also try the first two rows of the rotation mat, then gram-schmidt to get the third row.
        self.global_pose_delta = torch.zeros(1, 7, dtype=torch.float32, device=device)
        self.global_pose_delta[:, 3:] = torch.tensor(
            [1, 0, 0, 0], dtype=torch.float32, device=device
        )
        self.global_pose_delta = torch.nn.Parameter(self.global_pose_delta)

        # self.local_pose_deltas = torch.zeros(
        #     self.num_gaussians, 7, dtype=torch.float32, device=device
        # )
        # self.local_pose_deltas[:, 3:] = torch.tensor(
        #     [1, 0, 0, 0], dtype=torch.float32, device=device
        # )
        # self.local_pose_deltas = torch.nn.Parameter(self.local_pose_deltas)

        num_basis = 10
        self.local_pose_basis = torch.zeros(
            num_basis, 7, dtype=torch.float32, device=device
        )
        self.local_pose_basis[:, 3:] = torch.tensor(
            [1, 0, 0, 0], dtype=torch.float32, device=device
        )
        self.local_pose_basis = torch.nn.Parameter(self.local_pose_basis)
        self.local_pose_index = torch.rand(
            self.num_gaussians, num_basis, device=device
        )
        self.local_pose_index = torch.nn.Parameter(self.local_pose_index)

        # Optimizer.
        self.optimizer = torch.optim.Adamax(
            # [self.global_pose_delta, self.local_pose_deltas], lr=lr
            [self.global_pose_delta, self.local_pose_basis, self.local_pose_index], lr=lr
        )

        # Keep track of nearest neighbors for each gaussian.
        self.num_neighbors = num_neighbors
        model = NearestNeighbors(n_neighbors=(self.num_neighbors+1))
        model.fit(self.init_means)
        _, nearest_ids = model.kneighbors(self.init_means)
        nearest_ids = nearest_ids[:, 1:]  # remove self from nearest neighbors.
        self.nearest_ids = torch.Tensor(nearest_ids).to(device).int()

        # Note, blur seems to hurt for ARAP / per-gaussian optimization.
        # self.blur = torchvision.transforms.GaussianBlur(kernel_size=[21,21]).cuda()

        self.render_lock = render_lock

    def step(
        self,
        image: torch.Tensor,
        scale: float = 0.0,
        n_steps: int = 100,
    ) -> None:
        """Perform `n_steps` optimization steps, given target image."""
        with torch.no_grad():
            target_dino = self.get_dino_feat(image)

        # Take rigidity/affinity at the *canonical* frame (initial object position, init_means, aligned with GARField).
        # Small feature distance --> High affinity --> High rigidity.
        # Grouping features are on a unit sphere, so the max value for norm is actually 2.
        grouping_feats = get_affinity_at_scale(self.pipeline, self.init_means, scale)
        rigidity = (2 - (
            grouping_feats[self.nearest_ids] - grouping_feats[:, None, :]
        ).norm(dim=-1)) / 2
        rigidity = rigidity**3
        # rigidity = (rigidity - rigidity.min()) / (rigidity.max() - rigidity.min())
        # rigidity = (1 - (
        #     grouping_feats[self.nearest_ids] - grouping_feats[:, None, :]
        # ).norm(dim=-1)).clip(0, 1)
        # rigidity = 1/(1+torch.exp(-10*(rigidity-0.5)))

        # Update the rigidity based on the distance to the neighbors.
        # 1 / dist**2 (F = kx, U = 1/2 * k * x**2)
        # TODO plot a histogram of these distances?
        # dist_init = (self.init_means[self.nearest_ids] - self.init_means[:, None, :]).norm(dim=-1)
        # dist_median = dist_init.median()
        # dist_median = 0.01 # dist_init.float().quantile(0.9)
        # dist_median = dist_init[:, :10].float().quantile(0.9)
        # scale_factor = (1 / (1e-6 + (dist_init/dist_median)**2))
        # rigidity = rigidity * scale_factor

        # print(f"rigidity: {rigidity.min()}, {rigidity.max()}")
        # print("scaling", scale_factor.min(), scale_factor.max())

        for step in range(n_steps):
            self.optimizer.zero_grad()

            # First, let the global pose optimize.
            self.deform_gaussians(use_local = (step >= 30))

            with self.render_lock:
                self.pipeline.model.eval()
                outputs = self.pipeline.model.get_outputs(self.init_c2o)
            
            assert (
                "dino" in outputs.keys()
            ), "DINO feature not found in model outputs, perhaps object is out of frame?"
            assert type(outputs["dino"]) == torch.Tensor

            loss = 0

            # DINO recon loss: Pose should match, based on DINO features.
            output_dino = outputs["dino"]
            pix_loss = (output_dino - target_dino).abs().mean()  # L1 loss.

            # ARAP loss: Keep the gaussian local deformations small -- k*(delta_x)**2 small.
            dist = (
                self.gauss_params["means"][self.nearest_ids]
                - self.gauss_params["means"][:, None, :]
            ).norm(dim=-1)
            dist_init = (
                self.init_means[self.nearest_ids] - self.init_means[:, None, :]
            ).norm(dim=-1)
            delta_dist = (dist - dist_init) # = delta_x. [number of guassians, number of neighbors]

            # put a relu on the delta_dist, so that points can only pull, but not push.
            # you can only pull if your current distance > initial distance.
            # delta_dist = torch.nn.functional.leaky_relu(delta_dist)
            # delta_dist = torch.nn.functional.relu(delta_dist)

            # arap_loss = (0.1 * rigidity * (delta_dist**2)).mean(dim=-1).sum()  # sum over the neighbors.
            arap_loss = (rigidity * (delta_dist**2)).mean(dim=-1).sum()  # sum over the neighbors.

            # "Arap", but on the quaternions.
            # quat_norms = self.local_pose_deltas[:, 3:].norm(dim=-1, keepdim=True)
            # quat_normed = self.local_pose_deltas[:, 3:] / quat_norms
            # local_quat_loss = (rigidity * self.quatdist(
            #     quat_normed[self.nearest_ids], quat_normed[:, None, :]
            # )).mean(dim=-1).sum()

            # A point should have a similar motion basis, compared to its neighbors (especially if rigid).
            pose_index = torch.nn.functional.softmax(self.local_pose_index, dim=-1)
            local_quat_loss = (rigidity[:, :, None] * (
                pose_index[self.nearest_ids] - pose_index[:, None, :]
            )).abs().mean(dim=-1).mean(dim=-1).sum()  #* 0.0001

            loss = pix_loss + local_quat_loss
            loss.backward()
            self.optimizer.step()

    def get_dino_feat(self, rgb_frame: torch.Tensor) -> torch.Tensor:
        """Get the DINO feature of the frame, at the initial camera position.
        Args:
            rgb_frame: RGB frame tensor, in range [0...1].
        Returns:
            frame_pca_feats: DINO feature of the frame.
        """
        assert len(self.init_c2o.height.squeeze().shape) == 0
        height, width = self.init_c2o.height.item(), self.init_c2o.width.item()
        assert type(height) == int and type(width) == int

        rgb_frame = (
            resize(rgb_frame.permute(2, 0, 1), [height, width])
            .permute(1, 2, 0)
            .contiguous()
        )
        frame_pca_feats = self.pipeline.datamanager.dino_dataloader.get_pca_feats(
            rgb_frame.permute(2, 0, 1).unsqueeze(0), keep_cuda=True
        ).squeeze()
        frame_pca_feats = (
            resize(frame_pca_feats.permute(2, 0, 1), [height, width])
            .permute(1, 2, 0)
            .contiguous()
        )

        return frame_pca_feats

    def reset_transforms(self):
        """Reset the transforms of the gaussians to the initial values. In-place operation."""
        with torch.no_grad():
            self.gauss_params["means"] = self.init_means.clone()
            self.gauss_params["quats"] = self.init_quats.clone()

    def deform_gaussians(self, use_local: bool = True):
        """Apply the current pose deltas to the gaussians. In-place operation."""
        self.reset_transforms()

        # Apply local pose deltas.
        if use_local:
            # normalized_pose_index = self.local_pose_index / self.local_pose_index.sum(dim=-1, keepdim=True)
            normalized_pose_index = torch.nn.functional.softmax(self.local_pose_index, dim=-1)
            local_pose_deltas = torch.matmul(normalized_pose_index, self.local_pose_basis)
            quat_norms = local_pose_deltas[:, 3:].norm(dim=-1, keepdim=True)
            with torch.no_grad():
                self.gauss_params["quats"] = self.quatmul(
                    local_pose_deltas[:, 3:] / quat_norms, self.gauss_params["quats"]
                )
            self.gauss_params["means"] = (
                local_pose_deltas[:, :3]
                + torch.bmm(
                    quat_to_rotmat(local_pose_deltas[:, 3:] / quat_norms),
                    self.gauss_params["means"][..., None],
                ).squeeze()
            )

        # Apply global pose delta.
        quat_norms = self.global_pose_delta[:, 3:].norm(dim=-1, keepdim=True)
        with torch.no_grad():
            self.gauss_params["quats"] = self.quatmul(
                self.global_pose_delta[:, 3:] / quat_norms, self.gauss_params["quats"]
            )
        self.gauss_params["means"] = (
            self.global_pose_delta[:, :3]
            + torch.bmm(
                quat_to_rotmat(self.global_pose_delta[:, 3:] / quat_norms).expand(
                    self.num_gaussians, -1, -1
                ),
                self.gauss_params["means"][..., None],
            ).squeeze()
        )

    @staticmethod
    def quatmul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions, q0*q1."""
        w0, x0, y0, z0 = torch.unbind(q0, dim=-1)
        w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
        return torch.stack(
            [
                -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
                x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
            ],
            dim=-1,
        )

    @staticmethod
    def quatdist(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """Calculate the distance between two quaternions."""
        return 1 - (q0 * q1).sum(dim=-1)**2

    @staticmethod
    def reduce_pca(feats: torch.Tensor) -> torch.Tensor:
        """Reduce the DINO features [H, W, C_dino] to [H, W, 3]."""
        H, W, _ = feats.shape
        feats = feats.view(-1, feats.shape[-1])
        _, _, v = torch.pca_lowrank(feats, q=3)
        feats_pca = torch.matmul(feats, v)
        feats_pca = (feats_pca - feats_pca.min()) / (feats_pca.max() - feats_pca.min())
        return feats_pca.view(H, W, 3)

    def get_rgb_from_cam(self) -> torch.Tensor:
        """Get the rendered RGB image from the initial camera position."""
        self.pipeline.model.eval()
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs(self.init_c2o)
        assert ("rgb" in outputs.keys()) and (type(outputs["rgb"]) == torch.Tensor)
        return outputs["rgb"]


def compare_frames(frame1: torch.Tensor, frame2: torch.Tensor):
    """Compare two frames, plotting them side-by-side."""
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(frame1.cpu().numpy())
    axes[1].imshow(frame2.cpu().numpy())
    plt.savefig("foo.png")


if __name__ == "__main__":
    # Load pipeline, with garfield + DINO.
    config_path = Path("outputs/nerfgun2/dig/2024-04-11_140429/config.yml")
    # config_path = Path("outputs/nerfgun2/dig/2024-04-18_151706/config.yml")  # with scale reg
    # config_path = Path("outputs/garfield_plushie/dig/2024-04-17_155344/config.yml")
    viewer, pipeline = load_viewer_from_config(config_path)
    assert viewer.train_lock is not None

    # Crop the 3D model to the nerfgun only. Also, detach all the parameters, so they won't be "optimized".
    nerfgun_crop = np.load("nerfgun_crop.npy")
    # nerfgun_crop = np.load("nerfgun_crop_2024-04-18_151706.npy")
    for name in pipeline.model.gauss_params.keys():
        pipeline.model.gauss_params[name] = pipeline.model.gauss_params[name][
            nerfgun_crop
        ].detach()

    # Initialize the optimizer.
    optimizer = ARAPOptimizer(
        pipeline=pipeline, device=torch.device("cuda"), render_lock=viewer.train_lock
    )

    # Load video.
    video_path = Path("motion_vids/nerfgun_interact.MOV")
    video = cv2.VideoCapture(str(video_path.absolute()))

    # Optimize the pose of the gaussians, and save the frames.
    rgb_frames = []
    rgb_frames_overlay = []
    # for t in tqdm.tqdm(np.linspace(4.5, 6.0, 90)):
    for t in tqdm.tqdm(np.linspace(4.5, 5.0, 30)):
        frame = get_vid_frame(video, t)
        frame = torch.tensor(frame, dtype=torch.float32, device="cuda")
        optimizer.reset_transforms()
        optimizer.step(frame)

        viewer._trigger_rerender()

        rgb_frame = optimizer.get_rgb_from_cam()
        rgb_frames.append(rgb_frame.cpu().numpy() * 255)

        frame_interp = resize(
            frame.permute(2, 0, 1),
            [rgb_frame.shape[0], rgb_frame.shape[1]],
            antialias=True,
        ).permute(1, 2, 0)
        rgb_frame = rgb_frame * 0.9 + frame_interp * 0.1
        rgb_frames_overlay.append(rgb_frame.cpu().numpy() * 255)

    # Save the frames as a video.
    import moviepy.editor as mpy

    out_clip = mpy.ImageSequenceClip(rgb_frames, fps=10)
    out_clip.write_videofile("optimizer_rgb.mp4", fps=10)

    out_clip = mpy.ImageSequenceClip(rgb_frames_overlay, fps=10)
    out_clip.write_videofile("optimizer_rgb_overlay.mp4", fps=10)
