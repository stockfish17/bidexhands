# bidexhands 项目报告

## 本项目是基于 bidexhands 灵巧手为框架,新建了 XarmAllegroHandOver 方法为机械臂+灵巧手模型,同时复现了部分 dexpoint 论文中的算法<br>

## 使用方法:
运行指令与原 bidexhands 指令相同,但本实验中仅支持 --task=XarmAllegroHandOver, 算法上仅采取了 --algo=ppo(allegrohandover 类不支持多智能体的训练)<br>

## 实现细节:
1.使用了 assets/urdf/xarm_description/urdf/xarm6.urdf 的机械臂+灵巧手的配置文件作为机械臂+灵巧手的模型<br>
2.xarm_allegro_hand_over.py 的 setup_visual_obs_config 函数中实现了观测点云的特征拓展<br>
3.xarm_allegro_hand_over.py 的 setup_camera_from_config 中设置相机用于捕获点云<br>
4.xarm_allegro_hand_over.py 的 compute_reward 实现了论文 3.1 的接触算法,将奖励值 reward 作为返回结果<br>
5.支持了双臂独立驱动控制<br>
```python
self.agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
self.num_agents = 2
```
6.xarm_allegro_hand_over.py 的 _create_envs 函数中实现了机械臂+灵巧手 urdf 的加载,能够使用到点云处理中<br>
7.xarm_allegro_hand_over.py 的 compute_observations 函数中实现了深度图的获取并使用,并将点云数据结果存储到缓冲区中<br>
8.train.py 中实现了深度图转点云以及点云规范化<br>

## 待解决的问题:
1.相机设置的不合理,经常找不到点,可能通过多设置相机取点的方法解决<br>
2.触发随机生成的情况较多,没有完全成功复现点云算法<br>
```python
depth_image = np.random.rand(480, 640) * 0.5
```
3.尝试实现论文 3.2 想象点云的显式生成但没有成功,对于机器人运动学模型处理不到位,也导致了点云处理并不成功<br>
下面是 ppo 算法在 num_envs=2048, steps_num=100000000 下得到的效果图:<br>
![Image](./DexterousHands/bidexhands/logs/XarmAllegroHandOver/figure.png)
曲线波动比较大,需要进一步的完善.<br>

## 一些见解与体会
bidexhands 灵巧手需要掌握强化学习的知识,点云处理的方法,论文的复现以及项目本身的理解与解耦合等方法.作者在配置环境,项目理解与解耦合,点云处理上均遇到了一些困难,对于点云处理也尚未理解通透.这个项目也锻炼了我的搜索能力,代码复现以及项目深入理解,希望自己下一个项目可以做的更好.<br>