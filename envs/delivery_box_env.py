"""
택배상자 질량 중심 제어 Gymnasium 환경 (개선 버전)

개선 사항:
1. 보상 함수 연속화 + 착지 충격 패널티
2. 관측 설계 개선 (ID 기반 접근, 착지 예측 특징 추가)
3. 도메인 랜덤화 지원
4. 커리큘럼 학습 지원
5. XML 경로 유연화
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DomainRandomization:
    """도메인 랜덤화 설정"""
    enabled: bool = False

    # 질량 랜덤화 범위 (비율)
    mass_range: Tuple[float, float] = (0.9, 1.1)

    # 마찰 계수 랜덤화 범위 (비율)
    friction_range: Tuple[float, float] = (0.8, 1.2)

    # 액추에이터 출력 스케일 랜덤화 (비율)
    actuator_scale_range: Tuple[float, float] = (0.9, 1.1)

    # 센서 노이즈 표준편차
    sensor_noise_std: float = 0.01


@dataclass
class CurriculumConfig:
    """커리큘럼 학습 설정"""
    enabled: bool = False

    # 초기 낙하 높이 (점진적으로 증가)
    initial_drop_height: float = 1.0
    final_drop_height: float = 3.7

    # 초기 기울기 범위 (점진적으로 증가)
    initial_tilt_range: float = 5.0   # 도
    final_tilt_range: float = 15.0    # 도

    # 현재 진행률 (0.0 ~ 1.0)
    progress: float = 0.0


class DeliveryBoxEnv(gym.Env):
    """
    택배상자 질량 중심 제어 환경 (개선 버전)

    관측 공간 (19차원):
        - box_pos (3): 상자 위치 (x, y, z)
        - box_quat (4): 상자 자세 (쿼터니언)
        - box_linvel (3): 상자 선속도
        - box_angvel (3): 상자 각속도
        - mass_pos (2): 질량체 위치 (x, y) - 상자 기준
        - mass_vel (2): 질량체 속도
        - time_to_landing (1): 착지까지 예상 시간
        - normalized_height (1): 정규화된 높이

    행동 공간 (2차원):
        - motor_x (-1 ~ 1): X축 모터 제어
        - motor_y (-1 ~ 1): Y축 모터 제어

    보상 (연속형):
        - 착지 기울기: exp(-k * tilt) 형태의 연속 보상
        - 착지 충격: 착지 속도에 비례한 패널티
        - 제어 효율: 제어 입력 제곱에 비례한 패널티
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        drop_height: float = 3.7,
        max_episode_steps: int = 2000,
        control_frequency: int = 50,
        domain_randomization: Optional[DomainRandomization] = None,
        curriculum: Optional[CurriculumConfig] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.base_drop_height = drop_height
        self.max_episode_steps = max_episode_steps
        self.control_frequency = control_frequency

        # 도메인 랜덤화 및 커리큘럼 설정
        self.domain_rand = domain_randomization or DomainRandomization()
        self.curriculum = curriculum or CurriculumConfig()

        # MuJoCo 모델 로드 (경로 유연화)
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "models" / "delivery_box.xml"
        self.xml_path = Path(xml_path)

        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML 파일을 찾을 수 없습니다: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # 바디/조인트 ID 캐싱 (하드코딩 제거)
        self._cache_ids()

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

        # 관측 공간 (확장: 19차원)
        # box_pos(3) + box_quat(4) + box_linvel(3) + box_angvel(3)
        # + mass_pos(2) + mass_vel(2) + time_to_landing(1) + normalized_height(1)
        obs_dim = 19
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
        self.prev_action = np.zeros(2)
        self.landed = False
        self.landing_info = {}
        self.cumulative_control_cost = 0.0

        # 도메인 랜덤화 스케일 (에피소드별)
        self.actuator_scale = 1.0

    def _cache_ids(self):
        """바디/조인트 ID 캐싱 (하드코딩 제거)"""
        self.box_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "box_body"
        )
        self.joint_x_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_x"
        )
        self.joint_y_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_y"
        )

        # 조인트의 qpos/qvel 시작 인덱스
        self.joint_x_qpos_adr = self.model.jnt_qposadr[self.joint_x_id]
        self.joint_y_qpos_adr = self.model.jnt_qposadr[self.joint_y_id]
        self.joint_x_qvel_adr = self.model.jnt_dofadr[self.joint_x_id]
        self.joint_y_qvel_adr = self.model.jnt_dofadr[self.joint_y_id]

    def _init_viewer(self):
        """MuJoCo 뷰어 초기화"""
        try:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        except Exception:
            pass

    def _get_current_drop_height(self) -> float:
        """커리큘럼에 따른 현재 낙하 높이"""
        if not self.curriculum.enabled:
            return self.base_drop_height

        progress = self.curriculum.progress
        return (
            self.curriculum.initial_drop_height +
            progress * (self.curriculum.final_drop_height - self.curriculum.initial_drop_height)
        )

    def _get_current_tilt_range(self) -> float:
        """커리큘럼에 따른 현재 초기 기울기 범위 (라디안)"""
        if not self.curriculum.enabled:
            return np.radians(10)

        progress = self.curriculum.progress
        tilt_deg = (
            self.curriculum.initial_tilt_range +
            progress * (self.curriculum.final_tilt_range - self.curriculum.initial_tilt_range)
        )
        return np.radians(tilt_deg)

    def _get_obs(self) -> np.ndarray:
        """현재 관측값 반환 (개선: ID 기반 접근 + 착지 예측 특징)"""
        # 상자 상태 (ID 기반 접근)
        box_pos = self.data.xpos[self.box_body_id].copy()
        box_quat = self.data.xquat[self.box_body_id].copy()

        # 상자 속도 (freejoint의 qvel: 처음 6개)
        box_linvel = self.data.qvel[0:3].copy()
        box_angvel = self.data.qvel[3:6].copy()

        # 질량체 위치 (ID 기반 인덱싱)
        mass_x = self.data.qpos[self.joint_x_qpos_adr]
        mass_y = self.data.qpos[self.joint_y_qpos_adr]
        mass_pos = np.array([mass_x, mass_y])

        # 질량체 속도 (qvel에서 직접 가져오기)
        mass_vx = self.data.qvel[self.joint_x_qvel_adr]
        mass_vy = self.data.qvel[self.joint_y_qvel_adr]
        mass_vel = np.array([mass_vx, mass_vy])

        # 착지 예측 특징
        height = box_pos[2]
        vz = box_linvel[2]

        # 착지까지 예상 시간 (등가속도 운동 근사)
        # h = vz*t + 0.5*g*t^2, 단순화: t ≈ sqrt(2h/g) (vz가 음수일 때)
        if vz < 0 and height > 0.1:
            # 2차 방정식 풀이: 0.5*g*t^2 + vz*t - h = 0
            g = 9.81
            discriminant = vz**2 + 2*g*height
            if discriminant > 0:
                time_to_landing = (-vz + np.sqrt(discriminant)) / g
            else:
                time_to_landing = 0.0
        else:
            time_to_landing = max(0.0, height / max(abs(vz), 0.1))

        # 시간 정규화 (0~1 범위로, 최대 2초 기준)
        time_to_landing = np.clip(time_to_landing / 2.0, 0.0, 1.0)

        # 높이 정규화 (0~1 범위로, 최대 높이 기준)
        max_height = self._get_current_drop_height() + 0.5
        normalized_height = np.clip(height / max_height, 0.0, 1.0)

        # 센서 노이즈 추가 (도메인 랜덤화)
        if self.domain_rand.enabled and self.domain_rand.sensor_noise_std > 0:
            noise_std = self.domain_rand.sensor_noise_std
            box_pos += self.np_random.normal(0, noise_std, 3)
            box_linvel += self.np_random.normal(0, noise_std, 3)
            box_angvel += self.np_random.normal(0, noise_std, 3)
            mass_pos += self.np_random.normal(0, noise_std * 0.1, 2)

        obs = np.concatenate([
            box_pos,                    # 3
            box_quat,                   # 4
            box_linvel,                 # 3
            box_angvel,                 # 3
            mass_pos,                   # 2
            mass_vel,                   # 2
            [time_to_landing],          # 1
            [normalized_height],        # 1
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """추가 정보 반환"""
        box_pos = self.data.xpos[self.box_body_id]
        box_quat = self.data.xquat[self.box_body_id]

        euler = self._quat_to_euler(box_quat)

        return {
            "box_height": box_pos[2],
            "box_roll": euler[0],
            "box_pitch": euler[1],
            "box_yaw": euler[2],
            "tilt_degrees": np.degrees(np.sqrt(euler[0]**2 + euler[1]**2)),
            "landed": self.landed,
            "step_count": self.step_count,
            "cumulative_control_cost": self.cumulative_control_cost,
            "curriculum_progress": self.curriculum.progress if self.curriculum.enabled else None,
            **self.landing_info
        }

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """쿼터니언을 오일러 각도 (roll, pitch, yaw)로 변환"""
        w, x, y, z = quat

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        """
        보상 계산 (개선: 연속형 보상 + 충격 패널티 + 제어 비용)

        Returns:
            reward: 보상값
            terminated: 에피소드 종료 여부
            truncated: 시간 초과 여부
        """
        box_pos = self.data.xpos[self.box_body_id]
        box_quat = self.data.xquat[self.box_body_id]
        box_linvel = self.data.qvel[0:3]

        reward = 0.0
        terminated = False
        truncated = False

        height = box_pos[2]
        euler = self._quat_to_euler(box_quat)
        roll, pitch = euler[0], euler[1]
        tilt = np.sqrt(roll**2 + pitch**2)

        # ========== 1. 낙하 중 보상 ==========
        if height > 0.2:
            # 1-1. 수평 유지 보상 (연속형: exp 기반)
            # tilt=0 → 1.0, tilt=0.5rad(28도) → 0.08
            tilt_reward = np.exp(-5.0 * tilt)
            reward += 0.1 * tilt_reward

            # 1-2. 제어 입력 제곱 패널티 (에너지 효율)
            control_cost = 0.001 * np.sum(action**2)
            reward -= control_cost
            self.cumulative_control_cost += control_cost

            # 1-3. 제어 변화량 패널티 (부드러운 제어 유도)
            action_diff = np.sum((action - self.prev_action)**2)
            reward -= 0.0005 * action_diff

        # ========== 2. 착지 판정 ==========
        if height < 0.15 and not self.landed:
            self.landed = True

            # 착지 속도 (수직 성분)
            landing_vz = abs(box_linvel[2])
            landing_velocity = np.linalg.norm(box_linvel)

            # 착지 시 기울기
            tilt_rad = tilt
            tilt_degrees = np.degrees(tilt_rad)

            # ----- 2-1. 기울기 보상 (연속형) -----
            # r_tilt = A * exp(-k * tilt)
            # tilt=0 → 100, tilt=0.26rad(15도) → 27, tilt=0.79rad(45도) → 2
            A_tilt = 100.0
            k_tilt = 5.0
            tilt_reward = A_tilt * np.exp(-k_tilt * tilt_rad)

            # ----- 2-2. 충격 패널티 (연속형) -----
            # 자유낙하 3.7m의 이론 착지 속도: sqrt(2*g*h) ≈ 8.5 m/s
            # 정규화하여 패널티 계산
            expected_velocity = np.sqrt(2 * 9.81 * self._get_current_drop_height())
            normalized_impact = landing_vz / expected_velocity
            impact_penalty = 10.0 * normalized_impact  # 최대 약 10점 패널티

            # ----- 2-3. 성공 보너스 -----
            if tilt_degrees < 5:
                success_bonus = 20.0
            elif tilt_degrees < 15:
                success_bonus = 10.0
            else:
                success_bonus = 0.0

            # ----- 2-4. 실패 패널티 (45도 이상) -----
            if tilt_degrees >= 45:
                failure_penalty = 30.0
            else:
                failure_penalty = 0.0

            # 최종 착지 보상
            landing_reward = tilt_reward - impact_penalty + success_bonus - failure_penalty

            reward += landing_reward

            # 착지 정보 저장
            self.landing_info = {
                "landing_tilt_degrees": tilt_degrees,
                "landing_tilt_rad": tilt_rad,
                "landing_velocity": landing_velocity,
                "landing_vz": landing_vz,
                "landing_reward": landing_reward,
                "tilt_reward": tilt_reward,
                "impact_penalty": impact_penalty,
                "success_bonus": success_bonus,
                "failure_penalty": failure_penalty,
                "success": tilt_degrees < 15
            }

            terminated = True

        # ========== 3. 시간 초과 ==========
        if self.step_count >= self.max_episode_steps:
            truncated = True

        # ========== 4. 비정상 종료 ==========
        if height < -0.5 or height > 15 or tilt > np.pi:
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

        # 도메인 랜덤화 적용
        if self.domain_rand.enabled:
            self._apply_domain_randomization()

        # 커리큘럼에 따른 낙하 높이
        drop_height = self._get_current_drop_height()
        box_height = drop_height + 0.125

        # 위치 설정
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = box_height

        # 자세 설정 (커리큘럼에 따른 기울기 범위)
        if self.np_random is not None:
            max_tilt = self._get_current_tilt_range()
            roll = self.np_random.uniform(-max_tilt, max_tilt)
            pitch = self.np_random.uniform(-max_tilt, max_tilt)
            yaw = self.np_random.uniform(-np.pi, np.pi)
            quat = self._euler_to_quat(roll, pitch, yaw)
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.data.qpos[3:7] = quat

        # 질량체 초기 위치 (중앙)
        self.data.qpos[self.joint_x_qpos_adr] = 0.0
        self.data.qpos[self.joint_y_qpos_adr] = 0.0

        # 속도 초기화
        self.data.qvel[:] = 0.0

        # 상태 초기화
        self.step_count = 0
        self.prev_mass_pos = np.zeros(2)
        self.prev_action = np.zeros(2)
        self.landed = False
        self.landing_info = {}
        self.cumulative_control_cost = 0.0

        # 시뮬레이션 진행
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _apply_domain_randomization(self):
        """도메인 랜덤화 적용"""
        if not self.domain_rand.enabled:
            return

        # 액추에이터 스케일 랜덤화
        low, high = self.domain_rand.actuator_scale_range
        self.actuator_scale = self.np_random.uniform(low, high)

    def _euler_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """오일러 각도를 쿼터니언으로 변환"""
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
        """한 스텝 진행"""
        # 행동을 모터 토크로 변환
        action = np.clip(action, -1.0, 1.0)
        ctrl = action * 50.0 * self.actuator_scale

        # 제어 입력 적용
        self.data.ctrl[:] = ctrl

        # 시뮬레이션 진행
        for _ in range(self.control_steps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # 관측, 보상, 종료 조건 계산
        obs = self._get_obs()
        reward, terminated, truncated = self._compute_reward(action)
        info = self._get_info()

        # 이전 행동 저장 (부드러운 제어 계산용)
        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def set_curriculum_progress(self, progress: float):
        """커리큘럼 진행률 설정 (0.0 ~ 1.0)"""
        self.curriculum.progress = np.clip(progress, 0.0, 1.0)

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


def register_env():
    """환경을 Gymnasium에 등록"""
    gym.register(
        id="DeliveryBox-v1",
        entry_point="envs.delivery_box_env:DeliveryBoxEnv",
        max_episode_steps=2000,
    )
