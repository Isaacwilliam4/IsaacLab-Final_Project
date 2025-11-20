from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz, quat_rotate_inverse
import random
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.gen.omap")
import omni.kit.test
from isaacsim.asset.gen.omap.bindings import _omap
from isaacsim.asset.gen.omap.utils import compute_coordinates, generate_image, update_location
import numpy as np
import matplotlib.pyplot as plt

# import omni
# import omni.kit.app
# ext_manager = omni.kit.app.get_app().get_extension_manager()
# ext_manager.set_extension_enabled_immediate("isaacsim.asset.get.omap", True)

# from isaacsim.asset.gen.omap import _omap

def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())

@configclass
class LeatherbackPathPlanningEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 100.0
    action_spaces = {f"robot_{i}": 2 for i in range(1)}
    observation_spaces = {f"robot_{i}": 24 for i in range(1)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(1)}
    possible_agents = ["robot_0"]

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    robot_0: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")

    wall_0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object0",
        spawn=sim_utils.CuboidCfg(
            size=(20, 0.5, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 6.0, 1), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )

    wall_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object1",
        spawn=sim_utils.CuboidCfg(
            size=(20, 0.5, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -6.0, 1), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )

    wall_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )

    wall_3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object3",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )

    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    env_spacing = 30.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 10

    goal_reward_scale = 20

    block_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block_.*",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),  # base height above ground
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    num_blocks = 10

class LeatherbackPathPlanningEnv(DirectMARLEnv):
    cfg: LeatherbackPathPlanningEnvCfg

    def __init__(self, cfg: LeatherbackPathPlanningEnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs):

        self.block_offsets = self._sample_positions_grid(cfg.num_blocks)

        offset_idx = 0
        for i in range(cfg.num_blocks):
            block_id = f"block_{i}"
            setattr(cfg, block_id, cfg.block_cfg.replace(prim_path=f"/World/envs/env_.*/{block_id}"))
            res = self.block_offsets[offset_idx]
            x,y = res[0].item(), res[1].item()
            cfg.__dict__[block_id].init_state.pos = (x,y,0.5)
            offset_idx += 1

        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless
        self._om = _omap.acquire_omap_interface()
        self._throttle_dof_idx, _ = self.robots["robot_0"].find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.robots["robot_0"].find_joints(self.cfg.steering_dof_name)

        self._throttle_state = {robot_id:torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32) for robot_id in self.robots.keys()}
        self._steering_state = {robot_id:torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32) for robot_id in self.robots.keys()}

        self.env_spacing = self.cfg.env_spacing


        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "goal_reward",
            ]
        }

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                    "target": sim_utils.SphereCfg(
                        radius=0.2,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "arrow1": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(0.1, 0.1, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
                    ),
                },
        )
        self.target = VisualizationMarkers(marker_cfg)
        context = omni.usd.get_context()
        self._stage = context.get_stage()
        self._physx = omni.physx.get_physx_interface()

        # Create once, reuse
        self._omap_gen = _omap.Generator(self._physx, context.get_stage_id())
        self._omap_gen.update_settings(
            0.1,  # cell size
            1.0,   # occupied value
            0.0,   # free value
            0.5,   # unknown value
        )

        self.occ_cell_size = 0.1
        self.occ_min_bound = None

        # Set up matplotlib stuff, etc.
        plt.ion()
        self.occupancy_plot_im = None


    def _setup_scene(self):
        self.wall_0 = RigidObject(self.cfg.wall_0)
        self.wall_1 = RigidObject(self.cfg.wall_1)
        self.wall_2 = RigidObject(self.cfg.wall_2)
        self.wall_3 = RigidObject(self.cfg.wall_3)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # Setup rest of the scene
        self.robots = {}
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        for i in range(self.num_robots):
            self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
            self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]

        for i in range(self.cfg.num_blocks):
            block_cfg = self.cfg.__dict__[f'block_{i}']
            block = RigidObject(block_cfg)

        

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _draw(self):
        marker_pos = []
        marker_orientation = []
        marker_scales = []
        marker_indices = []

        device = self.goal.device if isinstance(self.goal, torch.Tensor) else "cpu"

        # ----------------- Goal sphere (index 0) -----------------
        goal_vis = torch.zeros(1, 3, device=device)
        goal_vis[:, :2] = self.goal.clone().to(device)
        marker_pos.append(goal_vis)

        # identity quaternion in xyzw: (0,0,0,1)
        marker_orientation.append(torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device))
        marker_scales.append(torch.ones(1, 3, device=device))     # sphere scale
        marker_indices.append(torch.zeros(1, device=device))      # 0 → sphere cfg

        # ----------------- Arrows along path (index 1) -----------------
        if hasattr(self, "plan_waypoints") and len(self.plan_waypoints) > 1:
            for i, wp in enumerate(self.plan_waypoints):
                if i == len(self.plan_waypoints) - 1:
                    break  # no segment from last waypoint

                # current and next waypoint (2D)
                p0_xy = torch.from_numpy(self.plan_waypoints[i]).float().to(device)
                p1_xy = torch.from_numpy(self.plan_waypoints[i + 1]).float().to(device)

                # put them in 3D (z=0)
                wp_vis = torch.zeros(1, 3, device=device)
                wp_vis[:, :2] = p0_xy  # position arrow at the start of the segment
                wp_vis[:, 2] = 1
                marker_pos.append(wp_vis)

                # direction vector and yaw
                delta = p1_xy - p0_xy
                dx, dy = delta[0], delta[1]
                yaw = torch.atan2(dy, dx)  # angle around z

                # build quaternion from (roll=0, pitch=0, yaw=yaw)
                # using your helper: get_quaternion_tuple_from_xyz(roll, pitch, yaw)
                roll = torch.tensor(0.0, device=device)
                pitch = torch.tensor(0.0, device=device)
                quat_tuple = get_quaternion_tuple_from_xyz(roll, pitch, yaw)

                # quat_tuple is expected to be (x, y, z, w)
                orientation = torch.tensor(
                    [[quat_tuple[0], quat_tuple[1], quat_tuple[2], quat_tuple[3]]],
                    device=device,
                )
                marker_orientation.append(orientation)

                # arrow length = segment length (scale X), keep Y,Z small
                seg_len = torch.norm(delta)
                marker_scales.append(
                    torch.tensor([[seg_len.item(), 1, 1]], device=device)
                )

                # index 1 → arrow cfg
                marker_indices.append(torch.ones(1, device=device))

        # ----------------- Stack and visualize -----------------
        marker_pos = torch.cat(marker_pos, dim=0)                  # (N, 3)
        marker_orientations = torch.cat(marker_orientation, dim=0) # (N, 4)
        marker_scales = torch.cat(marker_scales, dim=0)            # (N, 3)
        marker_indices = torch.cat(marker_indices, dim=0).long()   # (N,)

        self.target.visualize(
            marker_pos,
            marker_orientations,
            scales=marker_scales,
            marker_indices=marker_indices,
        )

    def _update_occupancy(self):
        origin = self.scene.env_origins[0].clone()
        origin[2] = 0.5 
        
        origin = (origin[0].item(), origin[1].item(), origin[2].item())

        x = origin[0]
        y = origin[1]

        # Fixed global box around env 0
        min_bound = (x - 10, y - 10, 0.1)
        max_bound = (x + 10, y + 10, 1.5)

        # remember for world->grid conversion
        self.occ_min_bound = min_bound

        self._omap_gen.set_transform(origin, min_bound, max_bound)
        self._omap_gen.generate2d()

        self._update_occupancy_map(self._omap_gen)

    def _update_occupancy_map(self, generator):
        # Get grid dimensions (carb.Int3: x=width, y=height, z=depth)
        dims = generator.get_dimensions()
        width, height, _ = int(dims.x), int(dims.y), int(dims.z)

        # Flattened buffer -> 2D array (height x width)
        buf = np.array(generator.get_buffer(), dtype=np.float32)
        if buf.size == 0 or width == 0 or height == 0:
            return

        grid = buf.reshape((height, width))
        grid = np.fliplr(grid)
        grid = np.flipud(grid)

        # ----- update robot position overlay -----
        if self.occ_min_bound is not None:
            # robot world position (env 0)
            root_pos = self.robots["robot_0"].data.root_pos_w[0].detach().cpu().numpy()
            goal_pos = self.goal.detach().cpu().numpy()
            rx_goal, ry_goal = goal_pos[0], goal_pos[1]
            rx, ry = root_pos[0], root_pos[1]

            min_x, min_y, _ = self.occ_min_bound
            cell = self.occ_cell_size

            gx = (rx - min_x) / cell
            gy = (ry - min_y) / cell
            gx_goal = (rx_goal - min_x) / cell
            gy_goal = (ry_goal - min_y) / cell

            # # clamp to grid bounds
            # gx = np.clip(gx, 0, width  - 1)
            # gy = np.clip(gy, 0, height - 1)

            self.goal_idx = [gx_goal, gy_goal]
            self.robot_idx = [gx, gy]
            self.grid = grid


    def _pre_physics_step(self, actions: dict) -> None:
        self._throttle_action = actions["robot_0"][:, 0].repeat_interleave(4).reshape((-1, 4)) * self.cfg.throttle_scale
        self._throttle_action = torch.clamp(self._throttle_action, -self.cfg.throttle_max, self.cfg.throttle_max)
        self._throttle_state["robot_0"] = self._throttle_action
        
        self._steering_action = actions["robot_0"][:, 1].repeat_interleave(2).reshape((-1, 2)) * self.cfg.steering_scale
        self._steering_action = torch.clamp(self._steering_action, -self.cfg.steering_max, self.cfg.steering_max)
        self._steering_state["robot_0"] = self._steering_action

    def _apply_action(self) -> None:
        for robot_id in self.robots.keys():
            # self._throttle_state[robot_id] = -5*torch.ones_like(self._throttle_state[robot_id], device=self.device)
            self.robots[robot_id].set_joint_velocity_target(self._throttle_state[robot_id], joint_ids=self._throttle_dof_idx)
            self.robots[robot_id].set_joint_position_target(self._steering_state[robot_id], joint_ids=self._steering_dof_idx)


    def relative_velocity(obj_vel_w, ref_rot_w):
        """Convert world-frame velocity into reference-frame coordinates.
        
        Args:
            obj_vel_w (torch.Tensor): (E, 3) linear velocity of object in world frame
            ref_rot_w (torch.Tensor): (E, 4) quaternion of reference orientation in world frame
        
        Returns:
            torch.Tensor: (E, 3) velocity in the reference's local frame
        """
        return quat_rotate_inverse(ref_rot_w, obj_vel_w)
    
    def _get_observations(self) -> dict:
        self._update_occupancy()
        self._draw()
        return {"robot_0": {"robot_pos":self.robot_idx, "grid":self.grid, "goal":self.goal_idx}}
    
    def _get_rewards(self) -> dict:
        return {"robot_0": torch.zeros(self.num_envs)}

    def _get_dones(self) -> tuple[dict, dict]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return {"robot_0": time_out}, {"robot_0": time_out}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        episode_lengths = self.episode_length_buf[env_ids].to(torch.float32).clone() + 1
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        robot_offset, self.goal = self._sample_positions_grid(2)

        origins = self.scene.env_origins[env_ids]  # (N, 3)

        robot_ids = list(self.robots.keys())

        for i, robot_id in enumerate(robot_ids):
            # Reset robot internals
            self.robots[robot_id].reset(env_ids)
            self.actions[robot_id][env_ids] = 0.0

            # Get default states for these envs
            joint_pos = self.robots[robot_id].data.default_joint_pos[env_ids]
            joint_vel = self.robots[robot_id].data.default_joint_vel[env_ids]
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids].clone()

            # Place robot
            default_root_state[:, :2] += origins[:, :2]
            default_root_state[:, :2] += robot_offset

            # Write to sim
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Draw markers & reset episode logs
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids] / episode_lengths)
            extras["Episode_Reward/"+key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
    
    def _sample_positions_grid(self, num_samples, grid_spacing=2.0):
        """
        Samples well-separated 2D positions per environment using a coarse meshgrid
        so blocks don't overlap. Super fast for large env counts.
        """
        device = "cuda:0"

        # Use grid_spacing >= min_dist to guarantee spacing
        xs = torch.arange(-8, 8, grid_spacing, device=device)
        ys = torch.arange(-4, 4, grid_spacing, device=device)
        xv, yv = torch.meshgrid(xs, ys, indexing="ij")

        grid_points = torch.stack([xv.flatten(), yv.flatten()], dim=-1)
        if hasattr(self, "block_offsets"):
            matches = (grid_points.unsqueeze(1) == self.block_offsets.unsqueeze(0)).all(dim=-1)

            # For each row in t0, check if it matches ANY row in t1
            exists_in_t1 = matches.any(dim=1)  # shape (128,)

            # Invert to keep only those NOT in t1
            grid_points = grid_points[~exists_in_t1]

        # randomly pick num_samples without replacement
        perm = torch.randperm(grid_points.shape[0], device=device)
        chosen = grid_points[perm[:num_samples]]

        return chosen


    @torch.no_grad()
    def _draw_grid_markers(self):
        """
        Draws green dots at every valid grid point for every environment
        stored in self.valid_points (populated by _sample_positions_grid).
        """
        device = self.device
        if not hasattr(self, "valid_points"):
            raise RuntimeError("Run _sample_positions_grid first to populate valid_points")

        pos_chunks = []
        idx_chunks = []
        marker_counter = 0

        for i, pts in enumerate(self.valid_points):
            z_col = torch.full((pts.shape[0], 1), 0.05, device=device)
            pos_i = torch.cat([pts, z_col], dim=1)

            pos_chunks.append(pos_i)
            idx_chunks.append(marker_counter + torch.arange(pts.shape[0],
                                                            device=device,
                                                            dtype=torch.long))
            marker_counter += pts.shape[0]

        marker_positions = torch.cat(pos_chunks, dim=0)      # (M, 3)
        marker_indices  = torch.cat(idx_chunks, dim=0)       # (M,)
        marker_scales   = 10 * torch.ones((marker_positions.shape[0], 3), device=device)

        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=device)
        marker_orientations[:, 0] = 1.0  # identity quaternion

        if not hasattr(self, "grid_markers"):
            markers = {f"grid_{i}": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ) for i in range(marker_counter)}

            grid_marker_cfg = VisualizationMarkersCfg(
                prim_path="/World/GridMarkers",
                markers=markers
            )
            self.grid_markers = VisualizationMarkers(grid_marker_cfg)

        self.grid_markers.visualize(
            marker_positions,
            marker_orientations,
            scales=marker_scales,
            marker_indices=marker_indices,
        )