from controller import Robot, Motor, Motion, LED, Camera, Gyro, Accelerometer, PositionSensor, GPS
import sys
sys.path.append('E:/Webots/Webots/projects/robots/robotis/darwin-op/libraries/python37')
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import BatchSampler, SubsetRandomSampler
from pathlib import Path
from gym import spaces
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse
import torch
import numpy as np
import gym
import cv2
import math
import time
import datetime
import random
import os
import warnings
import shutil

warnings.filterwarnings("ignore", category=Warning)
# 11.14

is_fall = False
robot = Robot()
res = 0


def process_img(image):
    image = image.astype(np.float32)
    image = cv2.resize(image, (84, 84))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image=cv2.threshold(image, 199, 1, cv2.THRESH_BINARY_INV)
    # 此时小于阈值的像素点置maxval，大于阈值的像素点置0；
    # return image[1]
    return image / 255.0


# Created by Zsq on 2023/11/14
# Description: 对图像进行预处理
# ------------------------------------------------------------------------------
def pre_processing(image):
    image = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    # _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    # 这行代码将处理后的图像返回作为函数的输出。
    # image[None, :, :]将图像的维度扩展为(1, height, width)，其中 None 表示新增的维度。
    # .astype(np.float32)将图像的数据类型转换为float32。
    return image[None, :, :].astype(np.float32)
    # return image / 255.0


# class arguments():
    # def __init__(self):
        # self.gamma = 0.99
        # self.action_dim = 3
        # self.obs_dim = (4, 84, 84)  # 80*80*3
        # self.capacity = 30000
        # self.cuda = 'cpu'
        # self.Frames = 4
        # self.episodes = int(5000)
        # self.updatebatch = 512
        # self.test_episodes = 10
        # self.epsilon = 0.1
        # self.Q_NETWORK_ITERATION = 50
        # self.learning_rate = 0.001


# Created by Zsq on 2023/11/14
# Description: 超参数设置
# ------------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of PPO to Basket robot""")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--log_path", type=str, default="runs")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.2)
    # parser.add_argument("--batch_size",type=int, default=2048)
    parser.add_argument("--batch_size",type=int, default=512)
    # parser.add_argument("--mini_batch_size",type=int, default=64)
    parser.add_argument("--mini_batch_size",type=int, default=16)

    args = parser.parse_args()
    return args


# Created by Zsq on 2023/11/14
# Description: 策略网络
# ------------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.Tanh())
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Sequential(nn.Linear(512, 3))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.flat(output)
        output = self.drop(output)
        output = self.fc1(output)
        return nn.functional.softmax(self.fc3(output), dim=1)


# Created by Zsq on 2023/11/14
# Description: 价值网络
# ------------------------------------------------------------------------------
class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, input):
        return self.net(input)


# Created by Zsq on 2023/11/14
# Description: 计算优势函数
# ------------------------------------------------------------------------------
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class ShotEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(160, 120, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.state = None
        self.done = False
        self.isSuccess = False
        self.shooting = Shooting()
        self.step_counter = 0
        self.kMaxEpisodeSteps = 10

    def get_observation(self):  # 获取相机图像返回值为可以直接传给state的形式
        image = self.shooting.camera.getImageArray()
        imgArray = np.array(image).astype(np.uint8)
        return imgArray

    def step(self, action):
        if action == 0:  # 投篮
            self.isSuccess = self.shooting.shot()
        elif action == 1:  # 左转一步
            self.shooting.left()
        elif action == 2:  # 右转一步
            self.shooting.right()
        self.step_counter += 1
        self.state = self.get_observation()
        reward = 0
        if action == 0 and self.isSuccess:
            reward = 100
            self.done = True
            global res
            res = res + 1
        elif action == 0 and not self.isSuccess:
            reward = -1
            self.done = True
        elif action == 1 or action == 2:
            reward = 0
            self.done = False
        if self.step_counter > self.kMaxEpisodeSteps:
            reward = -5
            self.done = True
        return self.state, reward, self.done, {}

    def reset(self):  # 重置智能体状态
        with open('E:/yan2/basket/basket_ppo/Basket/controllers/resetFlag.txt', 'r+') as file:
            file.write('0')
        self.step_counter = 0
        self.shooting.robot_reset()
        startTime = robot.getTime()
        while 2 + startTime >= robot.getTime():
            self.shooting.myStep()

        # a = self.robot.getPositionSensor('ShoulderLS')
        self.shooting.myStep()
        self.state = self.get_observation()
        self.done = False
        return self.state

    def takeTheBall(self):
        return self.shooting.takeTheBall()

    def find_basket(self):
        return self.shooting.find_basket()

    def up_head(self):
        return self.shooting.up_head()


class Shooting():
    def __init__(self):
        self.timestep = int(robot.getBasicTimeStep())
        self.gaitManager = RobotisOp2GaitManager(robot, 'config.ini')
        self.motionManager = RobotisOp2MotionManager(robot)

        self.gaitManager.setBalanceEnable(True)

        # --------------------------------启动传感器----------------------------------

        self.motors = []
        self.minMotorPositions = []
        self.maxMotorPositions = []

        motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                     'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
                     'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
                     'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head')

        self.left_extend = robot.getMotor('left_extend')
        self.right_extend = robot.getMotor('right_extend')
        self.eyeLed = robot.getLED('EyeLed')
        self.headLed = robot.getLED('HeadLed')
        self.camera = robot.getCamera('Camera')
        self.camera1 = robot.getCamera('new_camera')
        #  self.cameraData = self.camera.getImage()
        self.camera_F = 155.952
        self.accelerometer = robot.getAccelerometer('Accelerometer')
        self.gyro = robot.getGyro('Gyro')
        self.gps = robot.getGPS('gps')
        self.camera.enable(2 * self.timestep)
        self.camera1.enable(2 * self.timestep)
        self.accelerometer.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.gps.enable(self.timestep)

        self.fup = 0
        self.fdown = 0

        for i in range(len(motorName)):
            self.motors.append(robot.getMotor(motorName[i]))
            # 获取电机的最小最大位置
            self.minMotorPositions.append(self.motors[i].getMinPosition())
            self.maxMotorPositions.append(self.motors[i].getMaxPosition())
            # 启动传感器
            sensorName = motorName[i]
            sensorName = sensorName + 'S'
            # print(sensorName+'S')
            s = robot.getPositionSensor(sensorName).enable(self.timestep)

        # ---------------------------------启动结束-----------------------------------

    # myStep()函数，完成一个仿真步长的仿真。
    def myStep(self):
        ret = robot.step(self.timestep)
        if ret == -1:
            exit(0)

    # wait()函数，输入为等待毫秒数，使机器人等待一段时间。
    def wait(self, ms):
        startTime = robot.getTime()
        s = ms / 1000
        while s + startTime >= robot.getTime():
            self.myStep()

    # 通过加速度传感器获取机器人y轴的加速度值，当其值大于/小于某值一段时间后，
    # 判断机器人背部朝下/面部朝下摔倒，然后执行对应的起身动作。
    def isFall(self):
        ACC_TOLERANCE = 80.
        ACC_STEP = 20.

        acc = self.accelerometer.getValues()
        if acc[1] < 512.0 - ACC_TOLERANCE:
            self.fup += 1
        else:
            self.fup = 0

        if acc[1] > 512.0 + ACC_TOLERANCE:
            self.fdown += 1
        else:
            self.fdown = 0
        global is_fall
        if self.fup > ACC_STEP:
            self.motionManager.playPage(1)
            self.motionManager.playPage(10)
            self.motionManager.playPage(9)
            self.fup = 0
            is_fall = True
        elif self.fdown > ACC_STEP:
            self.motionManager.playPage(1)
            self.motionManager.playPage(11)
            self.motionManager.playPage(9)
            self.fdown = 0
            is_fall = True
        return is_fall

    def robot_reset(self):
        # 重置机器人关节位置
        self.myStep()
        self.gaitManager.stop()
        self.motors[18].setPosition(0.0)  # neck
        self.motors[19].setPosition(0.08)  # head 0.08
        self.motors[7].setPosition(-0.03)  # PelvYL
        self.motors[9].setPosition(0.0)  # PelvL
        self.motors[11].setPosition(1.15)  # LegUpperL
        self.motors[13].setPosition(-2.25)  # LegLowerL
        self.motors[15].setPosition(-1.22)  # AnkleL
        self.motors[17].setPosition(-0.043)  # FootL
        self.motors[6].setPosition(0.03)  # PelvYR
        self.motors[8].setPosition(0.0)  # PelvR
        self.motors[10].setPosition(-1.15)  # LegUpperR
        self.motors[12].setPosition(2.25)  # LegLowerR
        self.motors[14].setPosition(1.22)  # AnkleR
        self.motors[16].setPosition(0.043)  # FootR
        self.motors[1].setPosition(0.75)  # ShoulderL
        self.motors[3].setPosition(0.274)  # ArmUpperL
        self.motors[5].setPosition(-0.5)  # ArmLowerL
        self.left_extend.setPosition(0.0)  # left_extend
        self.motors[0].setPosition(-0.75)  # ShoulderR
        self.motors[2].setPosition(-0.274)  # ArmUpperR
        self.motors[4].setPosition(0.5)  # ArmLowerR
        self.right_extend.setPosition(0.0)  # right_extend
        # self.camera.disable()
        # self.gaitManager.setXAmplitude(0.0)
        # self.gaitManager.setAAmplitude(0.0)

        # self.gps.disable()
        # self.camera.enable(self.timestep)
        # print('重置机器人关节位置')

    def checkIfSuccess(self):
        print("开始检测是否进球")
        sum = 0  # 最多计数
        pnumber = 0  # 小球在篮筐范围内计数
        while True:
            ballPosition = getBallPosition()
            x = int(ballPosition[0] * 100)
            y = int(ballPosition[1] * 100)
            z = int(ballPosition[2] * 100)
            print('x = {} y = {} z = {}'.format(x, y, z))
            if x in range(3, 10) and y in range(30, 41) and z in range(-3, 4):
                pnumber += 1
                print(pnumber)
                if pnumber >= 2:
                    print("恭喜进球")
                    return True
            sum += 1
            # if sum >= 100:
            if sum >= 300:
                print("没有进球")
                return False
            self.myStep()

    def holdBall(self):  # 保持拿球状态
        self.motors[0].setPosition(0.521)
        self.motors[1].setPosition(-0.462)
        self.motors[2].setPosition(-0.68)
        self.motors[3].setPosition(0.77)
        self.motors[4].setPosition(0.533)
        self.motors[5].setPosition(0.525)
        self.motors[19].setPosition(0.9)
        self.left_extend.setPosition(-1.5)  # -1.83 防止穿模
        self.right_extend.setPosition(0.889)  # 0.889

    def takeTheBall(self):
        walkf = 0
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        self.myStep()

        self.headLed.set(0x00FF00)
        self.eyeLed.set(0x00FF00)
        self.motionManager.playPage(1)
        self.wait(200)

        px = 0.
        py = 0.

        motion1 = Motion('takeBall_left.motion')
        motion2 = Motion('takeBall_right.motion')
        holdBall_motion2 = Motion('holdBall2.motion')

        self.gaitManager.start()
        self.gaitManager.setBalanceEnable(True)

        i = 0
        while True:
            i += 1
            if i >= 2000:
                return [-1, -1], False
            isFall = self.isFall()
            if isFall:
                global is_fall
                is_fall = False
                return [-1, -1], False
            self.camera.saveImage('img.png', 50)
            x, y, w = getBallCenter()
            if x != -1 and y != -1:

                distance = getBallDistance(self.camera_F, w)
                x = 0.015 * (2.0 * x / width - 1.0) + px
                y = 0.015 * (2.0 * y / height - 1.0) + py
                px = x
                py = y
                neckPosition = clamp(-x, self.minMotorPositions[18], self.maxMotorPositions[18])
                headPosition = clamp(-y, self.minMotorPositions[19], self.maxMotorPositions[19])

                if distance < 0.120:
                    walkf = 0
                    if distance > 0.117:  # 如果distance在0.122和0.119之间，可以拿取篮球了
                        # isTakeBall = True
                        self.gaitManager.stop()
                        if x <= 0.0:
                            motion1.play()
                            # print('play motion1')
                        elif x > 0.0:
                            motion2.play()
                        self.wait(3000)
                        holdBall_motion2.play()
                        self.wait(1000)
                        robotPosition = self.gps.getValues()
                        robot_x = -robotPosition[2] * 100
                        robot_y = robotPosition[0] * 100
                        return [robot_x, robot_y], True
                    else:
                        walkf = 1  # 需要后退

                if walkf == 1:
                    self.gaitManager.setXAmplitude(-0.1)
                    self.gaitManager.step(self.timestep)
                else:
                    self.gaitManager.setXAmplitude(0.3)
                    self.gaitManager.setAAmplitude(neckPosition)
                    self.gaitManager.step(self.timestep)

                self.motors[18].setPosition(neckPosition)
                self.motors[19].setPosition(headPosition)

            else:  # 没看到小球
                self.isFall()
                self.gaitManager.setXAmplitude(0)
                self.gaitManager.setAAmplitude(0)
                self.gaitManager.step(self.timestep)
                headPosition = clamp(0.7 * math.sin(2.0 * robot.getTime()), self.minMotorPositions[19],
                                     self.maxMotorPositions[19])
                self.motors[19].setPosition(headPosition)
                neckPosition = clamp(0.35 * math.sin(robot.getTime()), self.minMotorPositions[18],
                                     self.maxMotorPositions[18])
                self.motors[18].setPosition(neckPosition)

            self.myStep()

    def up_head(self):
        self.motors[19].setPosition(0.8)
        self.wait(300)
        self.camera.saveImage('img.png', 50)

    def find_basket(self):

        model = torch.hub.load('.', 'custom', path='./runs/train/exp/weights/best.pt', source='local')
        # 对单张图片进行推理
        # model = torch.load('./detect/best.pt')
        img = './img.png'  # 图片路径
        # shutil.copy(img, './tt/img.png')
        results = model(img)
        # results.xyxy 是一个列表，每个元素是一个张量，张量的每一行表示一个检测框，格式为[xmin, ymin, xmax, ymax, confidence, class]
        for i, det in enumerate(results.xyxy):
            # 打印图片路径
            # print(f'Image {i}: {Path(img).name}')
            # 如果没有检测到任何目标，打印"No objects found"
            if len(det) == 0:
                print('No objects found')
            else:
                # 打印每个检测框的信息
                for *xyxy, conf, cls in det:
                    # 计算检测框的中心位置
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    x_rez = x_center / 160
                    # print(f'Class: {cls.item()}, Confidence: {conf.item()}, BBox Center: ({x_center}, {y_center})')
                    print(f'basket BBox X Center: {x_rez}')

    def left(self):  # 左转
        print("执行左转")
        self.motors[19].setPosition(0.9)
        self.gaitManager.start()
        self.myStep()
        self.gaitManager.setBalanceEnable(True)
        time_start = robot.getTime()
        time_now = time_start
        # print("time_start={}".format(time_start))
        while True:
            time_now = robot.getTime()
            # if time_now - time_start <= 0.3:
            if time_now - time_start <= 0.5:
                self.myStep()
                self.gaitManager.setXAmplitude(0.5)
                self.gaitManager.setAAmplitude(0.5)
                self.gaitManager.step(self.timestep)
                self.holdBall()
                time_now = robot.getTime()
            else:  # 否则停下来
                break

    def right(self):  # 右转
        print("执行右转")
        self.motors[19].setPosition(0.9)
        self.gaitManager.start()
        self.myStep()
        self.gaitManager.setBalanceEnable(True)
        time_start = robot.getTime()
        time_now = time_start
        # print("time_start={}".format(time_start))
        while True:
            time_now = robot.getTime()
            # if time_now - time_start <= 0.5:
            if time_now - time_start <= 0.3:
                self.myStep()
                self.gaitManager.setXAmplitude(0.5)
                self.gaitManager.setAAmplitude(-0.5)
                self.gaitManager.step(self.timestep)
                self.holdBall()
                time_now = robot.getTime()
            else:  # 否则停下来
                break

    def shot(self):
        print('开始投篮')
        self.motors[19].setPosition(0.9)
        self.gaitManager.start()
        self.gaitManager.setBalanceEnable(True)
        self.gaitManager.stop()
        self.motors[19].setPosition(0.9)
        holdBall_motion1 = Motion('holdBall.motion')  # 向左后方转动胳膊
        # print('开始执行holdmotion')
        self.motors[19].setPosition(0.9)
        holdBall_motion1.play()
        self.wait(4200)
        time_now1 = 0
        # isSuccess = self.checkIfSuccess()
        pnumber = 0
        while robot.step(16) != -1:
            # 对球在篮筐范围帧数计数
            ballPosition = getBallPosition()
            x = int(ballPosition[0] * 100)
            y = int(ballPosition[1] * 100)
            z = int(ballPosition[2] * 100)
            # print('x = {} y = {} z = {}'.format(x, y, z))
            if x in range(3, 10) and y in range(30, 41) and z in range(-3, 4):
                pnumber += 1

            self.motors[19].setPosition(0.9)
            time_now1 = time_now1 + 16
            a = robot.getMotor('ShoulderL')
            a.setPosition(10)
            a.setVelocity(13)
            if (time_now1) > 2300:
                print('投篮完毕')
                break
        # isSuccess = self.checkIfSuccess()
        if pnumber >= 3:
            isSuccess = True
        else:
            isSuccess = False
        return isSuccess


def clamp(value, min, max):
    if min > max:
        assert 0
        return value
    return min if value < min else max if value > max else value


def wait_reset(robot, s):
    startTime = float(time.time())
    # print(startTime)
    while startTime + s > float(time.time()):
        # print(float(time.process_time()))
        robot.step(int(robot.getBasicTimeStep()))


def getBallCenter():
    center_x = -1
    center_y = -1
    width = -1

    img = cv2.imread('img.png')

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # set the lower_hsv and upper_hsv
    lower_hsv = np.array((5, 100, 200))
    upper_hsv = np.array((25, 230, 255))
    # 图像切割
    bw_img = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        center_x = x + w / 2
        center_y = y + h / 2
        width = w
    return center_x, center_y, width


def getBallDistance(camera_F, width):
    distance = (0.04 * camera_F) / width
    return distance


def getBallPosition():
    ballPosition = [-100, -100, -100]
    # print('?')
    with open('E:/yan2/basket/basket_ppo/Basket/controllers/ballPosition.txt', 'r') as file:
        data = file.readlines()
        while len(data) == 0:
            data = file.readlines()
        # print(data)
        # if len(data):
        # ballPosition[0] = float(data[0].strip('\n'))
        # ballPosition[1] = float(data[1].strip('\n'))
        # ballPosition[2] = float(data[2].strip('\n'))

        ballPosition[0] = float(data[0].strip('\n'))
        ballPosition[1] = float(data[1].strip('\n'))
        ballPosition[2] = float(data[2].strip('\n'))
    # print(ballPosition)
    return ballPosition


# Created by Zsq on 2023/11/14
# Description: 训练主函数
# ------------------------------------------------------------------------------
def train(opt, env):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1993)
    else:
        torch.manual_seed(123)
    actor = PolicyNet().cuda()
    critic = ValueNet().cuda()

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=opt.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=opt.lr)
    writer = SummaryWriter(opt.log_path)
    replay_memory = []
    evaluate_num = 0
    evaluate_rewards = []
    iter = 0
    while iter < opt.num_iters:
        terminal = False
        episode_return = 0.0
        obs = env.reset()
        env.takeTheBall()
        ballPosition = getBallPosition()
        ball_y = int(ballPosition[1] * 100)
        if ball_y < 10:
            iter -= 1
            continue
        # 待优化，需要设置默认头的朝向
        env.up_head()
        # 同步obs---state
        image = env.get_observation()
        image = pre_processing(image)
        image = torch.tensor(image).cuda()
        state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
        while not terminal:
            # print(state.shape)
            # print(state)
            prediction = actor(state)
            action_dist = torch.distributions.Categorical(prediction)
            action_sample = action_dist.sample()
            action = action_sample.item()
            next_state, reward, terminal, _ = env.step(action)
            next_state = pre_processing(next_state)
            next_state = torch.tensor(next_state).cuda()
            next_state = torch.cat(tuple(next_state for _ in range(4)))[None, :, :, :]
            replay_memory.append([state, action, reward, next_state, terminal])
            state = next_state
            episode_return += reward

            if len(replay_memory) > opt.batch_size:
                print("---------开始进行参数更新----------")
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*replay_memory)
                states = torch.cat(state_batch, dim=0).cuda()
                actions = torch.tensor(action_batch).view(-1, 1).cuda()
                rewards = torch.tensor(reward_batch).view(-1, 1).cuda()
                dones = torch.tensor(terminal_batch).view(-1, 1).int().cuda()
                next_states = torch.cat(next_state_batch, dim=0).cuda()

                with torch.no_grad():
                    td_target = rewards + opt.gamma * critic(next_states) * (1 - dones)
                    td_delta = td_target - critic(states)
                    advantage = compute_advantage(opt.gamma, opt.lmbda, td_delta.cpu()).cuda()
                    old_log_probs = torch.log(actor(states).gather(1, actions)).detach()

                for _ in range(opt.epochs):
                    for index in BatchSampler(SubsetRandomSampler(range(opt.batch_size)), opt.mini_batch_size, False):
                        log_probs = torch.log(actor(states[index]).gather(1, actions[index]))
                        ratio = torch.exp(log_probs - old_log_probs[index])
                        surr1 = ratio * advantage[index]
                        surr2 = torch.clamp(ratio, 1 - opt.eps, 1 + opt.eps) * advantage[index]  # 截断
                        actor_loss = torch.mean(-torch.min(surr1, surr2))
                        critic_loss = torch.mean(
                            nn.functional.mse_loss(critic(states[index]), td_target[index].detach()))
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        actor_loss.backward()
                        critic_loss.backward()
                        actor_optimizer.step()
                        critic_optimizer.step()
                replay_memory = []

        iter += 1
        if (iter+1) % 10 == 0:
            evaluate_num += 1
            evaluate_rewards.append(episode_return)
            print("当前投中:{}次".format(res))
            print("evaluate_num:{} \t episode_return:{} \t".format(evaluate_num, episode_return))
            writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step= iter)
        if (iter+1) % 1000 == 0:
            actor_dict = {"net": actor.state_dict(), "optimizer": actor_optimizer.state_dict()}
            critic_dict = {"net": critic.state_dict(), "optimizer": critic_optimizer.state_dict()}
            # torch.save(actor_dict, "{}/robot_actor_good".format(opt.saved_path))
            # torch.save(critic_dict, "{}/robot_critic_good".format(opt.saved_path))

    writer.close()


if __name__ == '__main__':
    with open('E:/yan2/basket/basket_ppo/Basket/controllers/resetFlag.txt', 'r+') as file:
        file.write('1')

    env = ShotEnv()
    opt = get_args()
    train(opt, env)

