"""Remote-policy Isaac Sim deploy for G1 Dex1 against an Isaac-GR00T N1.7 server.

Same desktop responsibilities as unitree_lerobot's eval_g1_sim_remote.py:
- read sim images from local ZMQ image_server
- read sim state and publish actions through local DDS
- send observations to a GPU policy server (Isaac-GR00T PolicyServer, ZMQ/msgpack)

Run inside the ``unitree_lerobot`` conda env with this directory on PYTHONPATH:
    python eval_g1_sim_groot.py \
      --policy_server_address=127.0.0.1:5555 \
      --repo_id=RooibosT/g1_pick_redblock_dex1_sim_merged_107demo \
      --actions_per_chunk=16 --frequency=30 --arm=G1_29 --ee=dex1
"""

from dataclasses import asdict, dataclass, field
import logging
from multiprocessing.sharedctypes import SynchronizedArray
import os
from pathlib import Path
from pprint import pformat
import threading
import time

from gr00t_sim_policy_client import Gr00tSimPolicyClient
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging
import logging_mp
import numpy as np
import torch
import unitree_lerobot
from unitree_lerobot.eval_robot.make_robot import (
    process_images_and_observations,
    setup_image_client,
    setup_robot_interface,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.sim_savedata_utils import is_success, process_data_add
from unitree_lerobot.eval_robot.utils.utils import cleanup_resources, to_list, to_scalar


logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


@dataclass
class Gr00tEvalSimConfig:
    repo_id: str
    root: str = ""
    policy_server_address: str = "127.0.0.1:5555"
    actions_per_chunk: int = 16  # must match the served variant's action horizon (16 or 40)
    chunk_size_threshold: float = 0.5

    frequency: float = 30.0
    image_host: str = "127.0.0.1"

    arm: str = "G1_29"
    ee: str = "dex1"
    motion: bool = False
    headless: bool = False
    sim: bool = True
    visualization: bool = False
    save_data: bool = False
    task_dir: str = "./data"
    max_episodes: int = 1200
    rename_map: dict[str, str] = field(default_factory=dict)


def _execute_action(action: torch.Tensor, robot_interface, cfg: Gr00tEvalSimConfig):
    arm_ctrl = robot_interface["arm_ctrl"]
    arm_ik = robot_interface["arm_ik"]
    ee_shared_mem = robot_interface["ee_shared_mem"]
    arm_dof = robot_interface["arm_dof"]
    ee_dof = robot_interface["ee_dof"]

    action_np = action.cpu().numpy()
    arm_action = action_np[:arm_dof]
    tau = arm_ik.solve_tau(arm_action)
    arm_ctrl.ctrl_dual_arm(arm_action, tau)

    if cfg.ee:
        ee_action_start_idx = arm_dof
        left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
        right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]

        if isinstance(ee_shared_mem["left"], SynchronizedArray):
            ee_shared_mem["left"][:] = to_list(left_ee_action)
            ee_shared_mem["right"][:] = to_list(right_ee_action)
        elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
            ee_shared_mem["left"].value = to_scalar(left_ee_action)
            ee_shared_mem["right"].value = to_scalar(right_ee_action)

    return action_np


def eval_remote_policy(cfg: Gr00tEvalSimConfig, dataset: LeRobotDataset):
    logger_mp.info(f"Arguments: {cfg}")

    rerun_logger = RerunLogger() if cfg.visualization else None
    image_info = None
    remote_policy_client = None
    action_receiver_thread = None

    try:
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        (
            arm_ctrl,
            ee_shared_mem,
            arm_dof,
            ee_dof,
            sim_state_subscriber,
            sim_reward_subscriber,
            episode_writer,
            reset_pose_publisher,
        ) = (
            robot_interface[key]
            for key in [
                "arm_ctrl",
                "ee_shared_mem",
                "arm_dof",
                "ee_dof",
                "sim_state_subscriber",
                "sim_reward_subscriber",
                "episode_writer",
                "reset_pose_publisher",
            ]
        )

        image_client = image_info["image_client"]
        camera_config = image_info["camera_config"]
        expected_image_keys = set(dataset.meta.camera_keys)
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        # Read the initial pose / task from the parquet-backed table directly
        # instead of dataset[from_idx]: __getitem__ would decode camera frames,
        # which we neither need nor downloaded (download_videos=False).
        row = dataset.hf_dataset[int(from_idx)]
        init_arm_pose = (
            torch.as_tensor(row["observation.state"], dtype=torch.float32)[:arm_dof]
            .cpu()
            .numpy()
        )
        tasks_df = dataset.meta.tasks
        task = tasks_df.index[tasks_df["task_index"] == int(row["task_index"])][0]

        remote_policy_client = Gr00tSimPolicyClient(
            server_address=cfg.policy_server_address,
            fps=cfg.frequency,
            chunk_size_threshold=cfg.chunk_size_threshold,
            actions_per_chunk=cfg.actions_per_chunk,
        )
        remote_policy_client.start()
        action_receiver_thread = threading.Thread(
            target=remote_policy_client.receive_actions_loop,
            daemon=True,
        )
        action_receiver_thread.start()

        user_input = input("Enter 's' to initialize the robot and start the remote evaluation: ")
        idx = 0
        full_state = None
        reward_stats = {"reward_sum": 0.0, "episode_num": 0.0}

        if user_input.lower() != "s":
            return

        logger_mp.info("Initializing robot to starting pose...")
        tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
        robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
        time.sleep(1.0)

        logger_mp.info(
            f"Starting remote evaluation loop at {cfg.frequency} Hz against {cfg.policy_server_address}."
        )
        while True:
            if cfg.save_data and reward_stats["episode_num"] == 0:
                episode_writer.create_episode()

            loop_start_time = time.perf_counter()
            observation, current_arm_q = process_images_and_observations(
                image_client, camera_config, arm_ctrl
            )
            observation = {
                key: value
                for key, value in observation.items()
                if not key.startswith("observation.images.") or key in expected_image_keys
            }
            if expected_image_keys and not expected_image_keys.issubset(observation):
                logger_mp.warning(
                    "Expected camera frames are not all ready; skipping this control tick."
                )
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
                continue
            if current_arm_q is None:
                logger_mp.warning("Arm state is not ready; skipping this control tick.")
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
                continue

            left_ee_state = right_ee_state = np.array([])
            if cfg.ee:
                with ee_shared_mem["lock"]:
                    full_state = np.array(ee_shared_mem["state"][:])
                    left_ee_state = full_state[:ee_dof]
                    right_ee_state = full_state[ee_dof:]

            state_tensor = torch.from_numpy(
                np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
            ).float()
            observation["observation.state"] = state_tensor
            observation["task"] = task

            if remote_policy_client.ready_to_send_observation():
                remote_policy_client.send_observation(
                    dict(observation),
                    timestep=remote_policy_client.current_timestep(),
                )

            action = remote_policy_client.pop_action()
            action_np = None
            if action is not None:
                action_np = _execute_action(action, robot_interface, cfg)

            if cfg.save_data and action is not None:
                process_data_add(
                    episode_writer,
                    observation,
                    current_arm_q,
                    full_state,
                    action,
                    arm_dof,
                    ee_dof,
                )
                is_success(
                    sim_reward_subscriber,
                    episode_writer,
                    reset_pose_publisher,
                    remote_policy_client,
                    cfg,
                    reward_stats,
                    init_arm_pose,
                    robot_interface,
                )

            if cfg.visualization and action_np is not None:
                visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)

            idx += 1
            reward_stats["episode_num"] += 1
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    except Exception:
        logger_mp.exception("Remote evaluation crashed")
    finally:
        if remote_policy_client is not None:
            remote_policy_client.stop()
        if action_receiver_thread is not None:
            action_receiver_thread.join(timeout=2.0)
        if image_info:
            cleanup_resources(image_info)
        if "sim_state_subscriber" in locals() and sim_state_subscriber:
            sim_state_subscriber.stop_subscribe()
            logger_mp.info("SimStateSubscriber cleaned up")
        if "sim_reward_subscriber" in locals() and sim_reward_subscriber:
            sim_reward_subscriber.stop_subscribe()
            logger_mp.info("SimRewardSubscriber cleaned up")


@parser.wrap()
def eval_main(cfg: Gr00tEvalSimConfig):
    logging.info(pformat(asdict(cfg)))
    # unitree_lerobot resolves its URDF/mesh assets via CWD-relative paths
    # ("unitree_lerobot/eval_robot/assets/..."), assuming the process runs from
    # the repo dir that contains the package. Replicate that from anywhere.
    # (__path__, not __file__: unitree_lerobot is a namespace package)
    url_lerobot_root = Path(list(unitree_lerobot.__path__)[0]).resolve().parent
    os.chdir(url_lerobot_root)
    logging.info(f"Working directory set to {url_lerobot_root} (for URDF asset paths)")
    # The client only needs meta/ (camera keys, episode index, task string) and
    # the data/ parquet (initial arm pose) — skip the ~2GB videos/ download.
    # The GR00T policy server needs no dataset at all (stats live in the
    # checkpoint's processor/).
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root or None, download_videos=False)
    eval_remote_policy(cfg, dataset)
    logging.info("End of remote eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
