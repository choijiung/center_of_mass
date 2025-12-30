"""
택배상자 질량 중심 제어 Gymnasium 환경

3.7m 높이에서 낙하하는 택배상자가 바닥면으로 정확히 착지하도록
내부 질량체(CoreXY)를 제어하는 강화학습 환경
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class DeliveryBoxEnv(gym.Env):
    """
    택배상자 질량 중심 제어 환경

    관측 공간 (Observation Space):
        - box_pos (3): 상자 위치 (x, y, z)
        - box_quat (4): 상자 자세 (쿼터니언)
        - box_linvel (3): 상자 선속도
        - box_angvel (3): 상자 각속도
        - mass_pos (2): 질량체 위치 (x, y) - 상자 기준
        - mass_vel (2): 질량체 속도 (추정)

    행동 공간 (Action Space):
        - motor_x (-1 ~ 1): X축 모터 제어
        - motor_y (-1 ~ 1): Y축 모터 제어

    보상 (Reward):
        - 바닥면으로 정확히 착지할수록 높은 보상
        - 기울어진 상태로 착지하면 패널티
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        drop_height: float = 3.7,
        max_episode_steps: int = 2000,
        control_frequency: int = 50,  # Hz
    ):
        super().__init__()

        self.render_mode = render_mode
        self.drop_height = drop_height
        self.max_episode_steps = max_episode_steps
        self.control_frequency = control_frequency

        # MuJoCo 모델 로드
        model_path = Path(__file__).parent.parent / "models" / "delivery_box.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # 시뮬레이션 설정
        self.model.opt.timestep = 0.001  # 1ms
        self.control_steps = int(1.0 / (self.control_frequency * self.model.opt.timestep))

        # 행동 공간: 정규화된 모터 제어 (-1 ~ 1)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # 관측 공간
        # box_pos(3) + box_quat(4) + box_linvel(3) + box_angvel(3) + mass_pos(2) + mass_vel(2) = 17
        obs_dim = 17
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 렌더링 설정
        self.viewer = None
        self.renderer = None
        if self.render_mode == "human":
            self._init_viewer()

        # 에피소드 상태
        self.step_count = 0
        self.prev_mass_pos = np.zeros(2)
        self.landed = False
        self.landing_info = {}

    def _init_viewer(self):
        """MuJoCo 뷰어 초기화"""
        try:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        except Exception:
            pass

    def _get_obs(self) -> np.ndarray:
        """현재 관측값 반환"""
        # 상자 상태
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_body")
        box_pos = self.data.xpos[box_body_id].copy()
        box_quat = self.data.xquat[box_body_id].copy()

        # 상자 속도 (freejoint의 qvel)
        box_linvel = self.data.qvel[0:3].copy()
        box_angvel = self.data.qvel[3:6].copy()

        # 질량체 위치 (조인트 위치)
        joint_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_x")
        joint_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_y")

        # freejoint는 7개 자유도 (pos 3 + quat 4)를 차지
        mass_x = self.data.qpos[7 + self.model.jnt_qposadr[joint_x_id] - self.model.jnt_qposadr[joint_x_id]]
        mass_y = self.data.qpos[7 + 1]  # joint_y는 joint_x 다음

        mass_pos = np.array([self.data.qpos[7], self.data.qpos[8]])

        # 질량체 속도 (수치 미분)
        mass_vel = (mass_pos - self.prev_mass_pos) * self.control_frequency
        self.prev_mass_pos = mass_pos.copy()

        obs = np.concatenate([
            box_pos,      # 3
            box_quat,     # 4
            box_linvel,   # 3
            box_angvel,   # 3
            mass_pos,     # 2
            mass_vel,     # 2
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """추가 정보 반환"""
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_body")
        box_pos = self.data.xpos[box_body_id]
        box_quat = self.data.xquat[box_body_id]

        # 쿼터니언을 오일러 각도로 변환
        euler = self._quat_to_euler(box_quat)

        return {
            "box_height": box_pos[2],
            "box_roll": euler[0],
            "box_pitch": euler[1],
            "box_yaw": euler[2],
            "landed": self.landed,
            "step_count": self.step_count,
            **self.landing_info
        }

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """쿼터니언을 오일러 각도 (roll, pitch, yaw)로 변환"""
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def _compute_reward(self) -> Tuple[float, bool, bool]:
        """
        보상 계산

        Returns:
            reward: 보상값
            terminated: 에피소드 종료 여부
            truncated: 시간 초과 여부
        """
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_body")
        box_pos = self.data.xpos[box_body_id]
        box_quat = self.data.xquat[box_body_id]
        box_linvel = self.data.qvel[0:3]

        reward = 0.0
        terminated = False
        truncated = False

        # 현재 높이
        height = box_pos[2]

        # 기울기 계산 (쿼터니언에서)
        euler = self._quat_to_euler(box_quat)
        roll, pitch = euler[0], euler[1]
        tilt = np.sqrt(roll**2 + pitch**2)

        # 1. 수평 유지 보상 (낙하 중)
        if height > 0.2:  # 아직 공중에 있을 때
            # 기울기가 작을수록 보상
            tilt_reward = np.exp(-5 * tilt)
            reward += 0.1 * tilt_reward

            # 질량체 이동량에 대한 작은 패널티 (에너지 효율)
            mass_pos = np.array([self.data.qpos[7], self.data.qpos[8]])
            mass_penalty = -0.01 * np.sum(mass_pos**2)
            reward += mass_penalty

        # 2. 착지 판정
        if height < 0.15 and not self.landed:
            self.landed = True

            # 착지 속도
            landing_velocity = np.linalg.norm(box_linvel)

            # 착지 시 기울기 (바닥면 착지 = 기울기 0)
            landing_tilt = tilt

            # 착지 보상 계산
            # - 기울기가 0에 가까울수록 (바닥면 착지) 높은 보상
            # - 최대 보상: 100점 (완벽한 바닥면 착지)
            tilt_degrees = np.degrees(landing_tilt)

            if tilt_degrees < 5:  # 5도 이하: 거의 완벽한 착지
                landing_reward = 100.0 * (1 - tilt_degrees / 5)
            elif tilt_degrees < 15:  # 15도 이하: 양호한 착지
                landing_reward = 50.0 * (1 - (tilt_degrees - 5) / 10)
            elif tilt_degrees < 45:  # 45도 이하: 보통
                landing_reward = 20.0 * (1 - (tilt_degrees - 15) / 30)
            else:  # 45도 이상: 실패 (모서리/꼭지점 착지)
                landing_reward = -50.0

            reward += landing_reward

            # 착지 정보 저장
            self.landing_info = {
                "landing_tilt_degrees": tilt_degrees,
                "landing_velocity": landing_velocity,
                "landing_reward": landing_reward,
                "success": tilt_degrees < 15
            }

            terminated = True

        # 3. 시간 초과
        if self.step_count >= self.max_episode_steps:
            truncated = True

        # 4. 비정상 종료 (상자가 너무 많이 기울거나 벗어남)
        if height < 0 or height > 10 or tilt > np.pi:
            reward = -100
            terminated = True

        return reward, terminated, truncated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """환경 초기화"""
        super().reset(seed=seed)

        # MuJoCo 데이터 리셋
        mujoco.mj_resetData(self.model, self.data)

        # 초기 상태 설정
        # freejoint: [x, y, z, qw, qx, qy, qz]
        box_height = self.drop_height + 0.125  # 상자 중심 높이

        # 위치 설정
        self.data.qpos[0] = 0.0  # x
        self.data.qpos[1] = 0.0  # y
        self.data.qpos[2] = box_height  # z

        # 자세 설정 (쿼터니언: 약간의 랜덤 기울기)
        if self.np_random is not None:
            # 최대 ±10도 랜덤 기울기
            max_tilt = np.radians(10)
            roll = self.np_random.uniform(-max_tilt, max_tilt)
            pitch = self.np_random.uniform(-max_tilt, max_tilt)
            yaw = self.np_random.uniform(-np.pi, np.pi)

            # 오일러 -> 쿼터니언
            quat = self._euler_to_quat(roll, pitch, yaw)
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.data.qpos[3:7] = quat

        # 질량체 초기 위치 (중앙)
        self.data.qpos[7] = 0.0  # joint_x
        self.data.qpos[8] = 0.0  # joint_y

        # 속도 초기화
        self.data.qvel[:] = 0.0

        # 상태 초기화
        self.step_count = 0
        self.prev_mass_pos = np.zeros(2)
        self.landed = False
        self.landing_info = {}

        # 시뮬레이션 진행
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _euler_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """오일러 각도를 쿼터니언으로 변환 (w, x, y, z 순서)"""
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        한 스텝 진행

        Args:
            action: 정규화된 모터 제어 입력 [-1, 1]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 행동을 모터 토크로 변환
        action = np.clip(action, -1.0, 1.0)
        ctrl = action * 50.0  # 최대 토크 50N

        # 제어 입력 적용
        self.data.ctrl[:] = ctrl

        # 시뮬레이션 진행 (control_steps 만큼)
        for _ in range(self.control_steps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # 관측, 보상, 종료 조건 계산
        obs = self._get_obs()
        reward, terminated, truncated = self._compute_reward()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """렌더링"""
        if self.render_mode == "human":
            if self.renderer is not None:
                self.renderer.update_scene(self.data)
                return self.renderer.render()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self):
        """환경 종료"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Gymnasium 등록
def register_env():
    """환경을 Gymnasium에 등록"""
    gym.register(
        id="DeliveryBox-v0",
        entry_point="envs.delivery_box_env:DeliveryBoxEnv",
        max_episode_steps=2000,
    )
