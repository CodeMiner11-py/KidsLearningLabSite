import random
import time
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import csv

AI_MODEL = "DrivingModel4.1"
FOLDER = "models/"
MODEL_NAME = FOLDER + AI_MODEL
DATA = {}

REWARD_TABLE = []
REWARD_CSV_FILE = "reward_table.csv"
REWARD_STEP_COUNTER_FILE = "reward_step_counter.txt"
REWARD_BUFFER_SIZE = 1000
CSV_INITIALIZED = False
GLOBAL_STEP_COUNTER = 0

GLOBAL_START_TIME = time.time()


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d__%I_%M__%p")


def get_relative_time():
    return f"t={time.time() - GLOBAL_START_TIME:.3f}s"


def initialize_reward_csv():
    global CSV_INITIALIZED
    with open(REWARD_CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['NumberOfSteps', 'Action', 'RewardAmt'])
    CSV_INITIALIZED = True
    print(f"Initialized {REWARD_CSV_FILE}")


def load_step_counter():
    global GLOBAL_STEP_COUNTER
    try:
        with open(REWARD_STEP_COUNTER_FILE, 'r') as f:
            GLOBAL_STEP_COUNTER = int(f.read().strip())
        print(f"Loaded step counter: {GLOBAL_STEP_COUNTER}")
    except FileNotFoundError:
        GLOBAL_STEP_COUNTER = 0
        print("No step counter found, starting from 0")


def save_step_counter():
    with open(REWARD_STEP_COUNTER_FILE, 'w') as f:
        f.write(str(GLOBAL_STEP_COUNTER))


def flush_reward_table():
    global CSV_INITIALIZED

    if not CSV_INITIALIZED:
        initialize_reward_csv()

    if REWARD_TABLE:
        with open(REWARD_CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(REWARD_TABLE)
        print(f"Flushed {len(REWARD_TABLE)} reward entries to {REWARD_CSV_FILE}")
        REWARD_TABLE.clear()
        save_step_counter()


class GameObject:
    def __init__(self, x, y, width=50, height=50):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def overlaps(self, other):
        return (self.x < other.x + other.width and
                self.x + self.width > other.x and
                self.y < other.y + other.height and
                self.y + self.height > other.y)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


class Pedestrian(GameObject):
    def __init__(self, x, y, direction):
        super().__init__(x, y, 30, 40)
        self.direction = direction
        self.speed = 1

    def update(self):
        if self.direction == 'left':
            self.x -= self.speed
        else:
            self.x += self.speed

    def is_out_of_bounds(self):
        return self.x < 150 or self.x > 550


class DrivingSimulation:
    def __init__(self, speed_multiplier=1):
        self.speed_multiplier = speed_multiplier
        self.dt = 0.03

        self.road_left = 200
        self.road_right = 500
        self.road_width = 300
        self.lane_width = 50

        self.car = GameObject(225, 400, 50, 80)
        self.lane = 1
        self.lane_changing = False
        self.lane_change_progress = 0
        self.lane_change_direction = None
        self.lane_change_speed = 10

        self.height = 0
        self.speed = 0
        self.acceleration = 0
        self.max_speed = 10
        self.min_speed = -5

        self.warning = False
        self.warning_timer = 0

        self.speed_bump = GameObject(200, -300, 300, 20)
        self.cones = self._create_cones()
        self.pedestrians = []
        self.ped_crossing = GameObject(200, -1260, 300, 40)
        self.finish_line = GameObject(200, -1900, 300, 20)
        self.action = 0

        self.time_elapsed = 0
        self.last_ped_spawn = 0
        self.ped_spawn_interval = 3.25 / speed_multiplier
        self.ped_update_interval = 0.04 / speed_multiplier
        self.last_ped_update = 0
        self.accel_duration = 0.5
        self.accel_timer = 0

    def _create_cones(self):
        cones = []
        cone_heights = [-400, -500, -600, -700, -800, -900, -1000, -1500, -1600, -1700]
        for h in cone_heights:
            x = random.choice([250, 300, 350, 400, 450])
            cones.append(GameObject(x, h, 30, 30))
        return cones

    def spawn_pedestrian(self):
        if not self.cones:
            spawn_y = -300
        else:
            min_cone_y = min(cone.y for cone in self.cones)
            spawn_y = min_cone_y - 300

        choice = random.choice([200, 450])
        direction = 'right' if choice == 200 else 'left'
        ped = Pedestrian(choice, spawn_y, direction)
        self.pedestrians.append(ped)

    def update_pedestrians_movement(self):
        if self.time_elapsed - self.last_ped_update >= self.ped_update_interval:
            for ped in self.pedestrians[:]:
                ped.update()
                if ped.is_out_of_bounds():
                    self.pedestrians.remove(ped)
            self.last_ped_update = self.time_elapsed

    def update_motion(self):
        if self.accel_timer > 0:
            self.accel_timer -= self.dt
            if self.accel_timer <= 0:
                self.acceleration = 0

        self.speed += self.acceleration

        self.speed = max(self.min_speed, min(self.max_speed, self.speed))

        if self.acceleration == 0:
            decay = 0.05
            if abs(self.speed) < decay:
                self.speed = 0
            elif self.speed > 0:
                self.speed -= decay
            else:
                self.speed += decay

        if abs(self.speed) > 0.05:
            self.height += self.speed

            self.speed_bump.y += self.speed
            for cone in self.cones:
                cone.y += self.speed
            for ped in self.pedestrians:
                ped.y += self.speed
            self.ped_crossing.y += self.speed
            self.finish_line.y += self.speed

    def update_lane_change(self):
        if self.lane_changing:
            move_amount = 1 if self.lane_change_direction == 'right' else -1
            self.car.x += move_amount
            self.lane_change_progress += 1

            if self.lane_change_progress >= 25:
                self.lane += 0.5 if self.lane_change_direction == 'right' else -0.5
                self.lane_changing = False
                self.lane_change_progress = 0
                self.lane_change_direction = None

    def check_oob(self):
        if self.lane > 6:
            self.move_left()
            self.set_warning("danger: out of bounds, enabling lane assist")
        elif self.lane < 1:
            self.move_right()
            self.set_warning("danger: out of bounds, enabling lane assist")

        if self.height > 2000 and self.speed > 0:
            self.set_warning("danger: out of bounds, move back")
        elif self.height < -200 and self.speed < 0:
            self.set_warning("danger: out of bounds, move forward")

    def set_warning(self, message):
        if not self.warning:
            self.warning = True
            self.warning_timer = 1.0 / self.speed_multiplier

    def update_warning(self, dt):
        if self.warning:
            self.warning_timer -= dt
            if self.warning_timer <= 0:
                self.warning = False

    def update(self, dt):
        self.time_elapsed += dt

        if self.time_elapsed - self.last_ped_spawn >= self.ped_spawn_interval:
            self.spawn_pedestrian()
            self.last_ped_spawn = self.time_elapsed

        self.update_motion()
        self.update_lane_change()
        self.update_pedestrians_movement()
        self.check_oob()
        self.update_warning(dt)

    def move_left(self):
        if not self.lane_changing and self.lane > 1:
            self.lane_changing = True
            self.lane_change_direction = 'left'
            self.lane_change_progress = 0

    def move_right(self):
        if not self.lane_changing and self.lane < 6:
            self.lane_changing = True
            self.lane_change_direction = 'right'
            self.lane_change_progress = 0

    def accelerate(self):
        if self.height < 2000:
            self.acceleration = 0.2
            self.accel_timer = self.accel_duration

    def decelerate(self):
        if self.height > -200:
            self.acceleration = -0.2
            self.accel_timer = self.accel_duration

    def stop_acceleration(self):
        self.acceleration = 0

    def get_observation(self):
        ped_x = self.pedestrians[0].x if self.pedestrians else 350

        cone_lanes = [cone.x for cone in self.cones[:10]]
        while len(cone_lanes) < 10:
            cone_lanes.append(350)

        obs = np.array([self.height, self.lane, ped_x] + cone_lanes, dtype=np.float32)

        obs[0] = np.clip((self.height - 900) / 900, -1, 1)
        obs[1] = np.clip((self.lane - 3.5) / 3.5, -1, 1)
        obs[2:] = np.clip((np.array(obs[2:]) - 350) / 150, -1, 1)

        return obs

    def is_done(self):
        return self.height >= 1900

    def calculate_reward(self):
        reward = 0

        reward += self.height / 1000

        if abs(self.speed) < 0.1:
            reward -= 0.5

        if self.lane >= 5.5:
            reward -= 0.5
        elif self.lane <= 1.5:
            reward -= 0.5

        for cone in self.cones:
            if self.car.overlaps(cone):
                reward -= 10
        for ped in self.pedestrians:
            if self.car.overlaps(ped):
                reward -= 10

        if self.lane_changing:
            reward -= 0.1

        if self.car.overlaps(self.speed_bump) and self.speed > 4:
            reward -= self.speed / 2

        if self.warning:
            reward -= 2

        if self.action == 0:
            reward -= 2

        return reward


class DrivingEnvironment(gym.Env):
    def __init__(self, speed_multiplier=1):
        super().__init__()
        self.speed_multiplier = speed_multiplier

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32
        )

        self.sim = None
        self.steps_since_last_update = 0
        self.update_interval = 30

        self.steps_since_accel = 0
        self.last_height = 0
        self.steps_without_progress = 0
        self.max_steps_without_progress = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = DrivingSimulation(self.speed_multiplier)
        self.steps_since_last_update = 0
        self.steps_since_accel = 0
        self.last_height = 0
        self.steps_without_progress = 0
        return self.sim.get_observation(), {}

    def step(self, action):
        global GLOBAL_STEP_COUNTER

        if action == 1:
            self.sim.accelerate()
            self.steps_since_accel = 0
        elif action == 2:
            self.sim.decelerate()
        elif action == 3:
            self.sim.move_left()
        elif action == 4:
            self.sim.move_right()

        self.sim.action = action
        self.steps_since_accel += 1
        GLOBAL_STEP_COUNTER += 1

        for _ in range(10):
            self.sim.update(self.sim.dt)

        observation = self.sim.get_observation()
        reward = self.sim.calculate_reward()

        height_change = abs(self.sim.height - self.last_height)
        if height_change < 1.0:
            self.steps_without_progress += 1
        else:
            self.steps_without_progress = 0
        self.last_height = self.sim.height

        if action != 1 and self.steps_since_accel > 100:
            reaction_penalty = min((self.steps_since_accel - 100) * 0.001, 5.0)
            reward -= reaction_penalty
        else:
            reaction_penalty = 0

        stuck = self.steps_without_progress >= self.max_steps_without_progress

        done = self.sim.is_done() or stuck

        if stuck:
            reward -= 50

        REWARD_TABLE.append([GLOBAL_STEP_COUNTER, action, round(reward, 4)])

        if len(REWARD_TABLE) >= REWARD_BUFFER_SIZE:
            flush_reward_table()

        str_reward = f"{reward:.4f}"
        reward_text = f"Reward {'+' if reward >= 0 else ''}{str_reward}"
        print(f"Action {action} Height {self.sim.height:.1f} {reward_text}")

        if reaction_penalty > 0:
            print(f"  └─ Reaction penalty: -{reaction_penalty:.3f} (no accel for {self.steps_since_accel} steps)")

        DATA[get_timestamp()] = str_reward

        if done:
            if stuck:
                print(
                    f"Episode FAILED! Stuck at height {self.sim.height:.1f} (no progress for {self.steps_without_progress} steps)")
            else:
                print(f"Episode complete! Height: {self.sim.height:.1f}")

        return observation, reward, done, False, {}


class PPOTrainer:
    def __init__(self, speed_multiplier=500, model_name=MODEL_NAME):
        self.speed_multiplier = speed_multiplier
        self.model_name = model_name
        self.model = None
        self.env = None
        self.nonvecenv = None

    def create_env(self):
        env = DrivingEnvironment(self.speed_multiplier)
        self.nonvecenv = env
        self.env = DummyVecEnv([lambda: env])
        return self.env

    def load_or_create_model(self):
        try:
            print(f"Loading model from {self.model_name}...")
            self.model = PPO.load(self.model_name)
            self.model.set_env(self.env)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Could not load model ({e}), creating new one...")
            self.model = PPO("MlpPolicy", self.env, verbose=1)

    def save_model(self, path=None):
        save_path = path or self.model_name
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

    def train(self, total_timesteps=10000, checkpoint_freq=1000000):
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=FOLDER + "callbacks/",
            name_prefix=AI_MODEL
        )

        epoch = 0

        while True:
            epoch += 1
            print(f"\n{'=' * 60}")
            print(f"Training Epoch {epoch} - {total_timesteps} timesteps")
            print(f"{'=' * 60}")

            try:
                self.model.learn(
                    total_timesteps=total_timesteps,
                    callback=checkpoint_callback,
                    reset_num_timesteps=False
                )
                print(f"Epoch {epoch} complete!")

                flush_reward_table()

                with open('Data.txt', 'w') as f:
                    to_add = ["Timestamp,Reward\n"]
                    for item in DATA:
                        to_add.append(f"{item},{DATA[item]}\n")
                    for item in to_add:
                        f.write(item)

                self.save_model()

            except KeyboardInterrupt:
                print("\nTraining interrupted by user!")
                flush_reward_table()
                self.save_model()
                break
            except Exception as e:
                print(f"Error during training: {e}")
                flush_reward_table()
                self.save_model()
                raise


if __name__ == "__main__":
    print(f"{get_relative_time()} Starting training...")

    load_step_counter()

    trainer = PPOTrainer(speed_multiplier=500, model_name=MODEL_NAME)
    trainer.create_env()
    trainer.load_or_create_model()

    try:
        trainer.train(total_timesteps=500000)
    except Exception as e:
        print(f"Training error: {e}")
        flush_reward_table()
        trainer.save_model()
    finally:
        flush_reward_table()
        save_step_counter()
        print("Training session ended.")