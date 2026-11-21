# 60-UAV Search 离线恢复可行性验证

OFFLINE_SEARCH_RECOVERY_FEASIBILITY = INCONCLUSIVE

物理恢复可行性 COMPLETE；可观测恢复可部署性 INCONCLUSIVE。

## 环境与范围

Git root: <local-home>/MARVEL_INTEGRATION；项目目录: <local-home>/MARVEL_INTEGRATION/MARVEL-main (1)/MARVEL-main
分支 extended-scalability；HEAD 5143fbdc4bdadf907f59c433e81ab11b2e98c6dc
只调用生产 KinematicSimple、ObstacleManager 和单机 SafetyShield；没有实例化 SimulationRuntime、scheduler、adapter 或策略，没有加载 checkpoint、训练或运行 fleet 仿真。
旧文件全部保留；本轮文件仅新增于 artifacts/offline_search_recovery_20261009/。

## 真实执行链

policy_adapter.py:242–284 产生 base actions 后调用 scheduler.apply；Search override 在 scheduler.py:1299–1381 调用 observed_next_hop。BFS 从 current_index 的图节点出发，选择 reachable 节点中最近 heat 的目的节点，返回图上的下一跳，直接读取图 coords，不经过 PolicyNet 360-node window decode。
Extended _override_heading 使用连续 atan2（scheduler.py:986–1023）；native 的角度量化和 post-selection 未应用于 extended。apply 末尾仍有 hazard/Safety guard；本案 trace 最终 goal 与 hop 坐标相同，没有目标替换证据。
harness 每 mission 保持同一动作 10 ticks；runtime.step:430–480 先调用 SafetyShield，再用 KinematicSimple 计算运动提案，再以 radius=0.2 调用 check_collision。拒绝时 continue，不提交 position/heading/velocity。SafetyShield 的静态判断只检查目标点，不检查沿途（safety_shield.py:38–51）。
KinematicSimple 平移沿实际位置→目标，速度1m/s、dt=0.1s，每tick最大0.1m；平移与 current/desired heading 无关。六种当前heading的独立检查得到相同提案。yaw_rate=35。
地图 maps_test/1.png，500x500、origin=(0,0)、cell_size=1m；碰撞规则为中心+四个轴向0.2m偏移五探针，格坐标用floor。这里验证的是现行规则，不是精确圆盘的Minkowski检测。

## 精度与缺失

CSV坐标4位小数、trace3位小数。历史 current heading、逐tick heading、完整belief/visible_cells、图coords/邻接、其他UAV位置/目标和原始地图hash均缺失。theta_current=0只是离线参数，未冒充历史heading。目标heading可由记录位置和goal按extended原语计算；历史姿态轨迹不可精确恢复。
本轮地图hash、CSV和生产源码hash保存在final/results.json，验证时一致；缺原地图hash，因此不能声称字节级历史一致。四个mission重放的位置与原CSV相差均<0.1mm，拒绝计数与trace一致。热区anchor从位置/距离拟合并核验整数格，属于推算公开anchor，不是读取隐藏survivor。

## UAV 43 / heat 0

停滞点 (149.6404, 74.7832)；推算heat (263.0000, 45.0000)，最大距离残差 8.897e-5m。

| mission | 起点 | 图current node | next-hop | 目标heading（重建） | 接受/拒绝tick | 终点误差 |
|---|---|---|---|---:|---:|---:|
| 42 | (149.3311, 74.5295) | (148.0000, 72.0000) | (156.0000, 80.0000) | 39.362036° | 4/6 | 4.044e-5m |
| 43 | (149.6404, 74.7832) | (152.0000, 76.0000) | (156.0000, 80.0000) | 39.362137° | 0/10 | 0.000e+0m |

- step 42 第5 tick首次拒绝：(149.6404, 74.7832) → (149.7177, 74.8466)，grid_obstacle；命中offset (0.0000, 0.2000)、格(col,row) (149.0000, 75.0000)、世界格盒 [149,75,150,76]。当前点与goal均合法。
- step 43 第1 tick首次拒绝：(149.6404, 74.7832) → (149.7177, 74.8466)，grid_obstacle；命中offset (0.0000, 0.2000)、格(col,row) (149.0000, 75.0000)、世界格盒 [149,75,150,76]。当前点与goal均合法。

精确机制：step42图节点→hop合法，但真实位置→hop不合法。step43映射到图节点(152,76)，真实UAV却仍在障碍另一侧；该图边合法，真实位置→图节点及→hop都不合法。因此仅过滤图边无法修复实际接入问题。

8方向中有5个合法0.1m首步：[-1,-1], [-1,0], [0,-1], [1,-1], [1,0].

### rejoin_original_hop_path

(149.6404, 74.7832) → (150.1404, 74.7832) → (155.1404, 79.7832) → (155.6404, 79.7832) → (156.0000, 80.0000)

| 动作 | target | target heading | 保持mission数 | physics ticks |
|---|---|---:|---:|---:|
| 1 | (150.1404, 74.7832) | 0.000000° | 1 | 10 |
| 2 | (155.1404, 79.7832) | 45.000000° | 8 | 80 |
| 3 | (155.6404, 79.7832) | 0.000000° | 1 | 10 |
| 4 | (156.0000, 80.0000) | 31.085443° | 1 | 10 |

路径长 8.490966m，总110ticks；heat距离 117.206817m → 112.578861m。每个动作整mission保持，不在tick中途换目标。到目标后剩余tick悬停是现有模型行为。所有落点与连续段均满足现行碰撞规则。

### heatward_exit_path

(149.6404, 74.7832) → (151.1404, 74.7832) → (151.6404, 74.2832)

| 动作 | target | target heading | 保持mission数 | physics ticks |
|---|---|---:|---:|---:|
| 1 | (151.1404, 74.7832) | 0.000000° | 2 | 20 |
| 2 | (151.6404, 74.2832) | 315.000000° | 1 | 10 |

路径长 2.207107m，总30ticks；heat距离 117.206817m → 115.145414m。每个动作整mission保持，不在tick中途换目标。到目标后剩余tick悬停是现有模型行为。所有落点与连续段均满足现行碰撞规则。

override未选退出动作的原因：总是直接执行非空BFS hop，没有真实位置接入段检查、拒绝反馈驱动的blocked transition记忆或局部绕行前缀。只有hop返回None才保留原MARVEL探索动作；本案始终override。

## UAV 0 / heat 1

停滞点 (124.7546, 75.2119)；推算heat (28.0000, 474.0000)，最大距离残差 8.127e-5m。

| mission | 起点 | 图current node | next-hop | 目标heading（重建） | 接受/拒绝tick | 终点误差 |
|---|---|---|---|---:|---:|---:|
| 55 | (124.3992, 75.5637) | (124.0000, 76.0000) | (128.0000, 72.0000) | 315.296693° | 5/5 | 8.467e-5m |
| 56 | (124.7546, 75.2119) | (124.0000, 76.0000) | (128.0000, 72.0000) | 315.297244° | 0/10 | 0.000e+0m |

- step 55 第6 tick首次拒绝：(124.7546, 75.2120) → (124.8257, 75.1416)，grid_obstacle；命中offset (0.2000, 0.0000)、格(col,row) (125.0000, 75.0000)、世界格盒 [125,75,126,76]。当前点与goal均合法。
- step 56 第1 tick首次拒绝：(124.7546, 75.2119) → (124.8257, 75.1416)，grid_obstacle；命中offset (0.2000, 0.0000)、格(col,row) (125.0000, 75.0000)、世界格盒 [125,75,126,76]。当前点与goal均合法。

精确机制：真实位置→hop，以及图节点(124,76)→(128,72)的边都不满足radius0.2规则。存在绕路到hop，但到hop后heat距离增加；图邻接不是物理可通行证明。

8方向中有5个合法0.1m首步：[-1,-1], [-1,0], [-1,1], [0,-1], [0,1].

### rejoin_original_hop_path

(124.7546, 75.2119) → (124.7546, 74.7119) → (127.2546, 72.2119) → (128.0000, 72.0000)

| 动作 | target | target heading | 保持mission数 | physics ticks |
|---|---|---:|---:|---:|
| 1 | (124.7546, 74.7119) | 270.000000° | 1 | 10 |
| 2 | (127.2546, 72.2119) | 315.000000° | 4 | 40 |
| 3 | (128.0000, 72.0000) | 344.130774° | 1 | 10 |

路径长 4.810468m，总60ticks；heat距离 410.357650m → 414.251132m。每个动作整mission保持，不在tick中途换目标。到目标后剩余tick悬停是现有模型行为。所有落点与连续段均满足现行碰撞规则。

### heatward_exit_path

(124.7546, 75.2119) → (124.7546, 76.7119) → (124.2546, 77.2119)

| 动作 | target | target heading | 保持mission数 | physics ticks |
|---|---|---:|---:|---:|
| 1 | (124.7546, 76.7119) | 90.000000° | 2 | 20 |
| 2 | (124.2546, 77.2119) | 135.000000° | 1 | 10 |

路径长 2.207107m，总30ticks；heat距离 410.357650m → 408.296148m。每个动作整mission保持，不在tick中途换目标。到目标后剩余tick悬停是现有模型行为。所有落点与连续段均满足现行碰撞规则。

override未选退出动作的原因：总是直接执行非空BFS hop，没有真实位置接入段检查、拒绝反馈驱动的blocked transition记忆或局部绕行前缀。只有hop返回None才保留原MARVEL探索动作；本案始终override。

## 有界搜索与独立复核

0.5m分辨率、以记录真实位置为中心的8邻接网格、12m半径、最多6000扩展。两机到原hop分别扩展23/7状态，局部heat距离下降至少2m分别扩展40/42状态。不声称全局最优或全局可达。
连续检查枚举五探针所有整数格边界的跨越参数，在边界、两侧和各常格区间内部调用真实check_collision；不是只查离散落点。限于本次无圆形/动态障碍的栅格场景。
四条路径名义执行共230ticks。verify_results.py独立重放名义起点和四个±0.00005m舍入角点，固定原waypoint/heading，全部无碰撞且到达末点；单UAV SafetyShield保持启用。四角检查仅为敏感性检查，不是任意状态不确定性的完整证明。
其他59个UAV状态缺失，因此无法验证真实fleet proximity条件。

## Oracle与可部署性

真值地图仅作为诊断oracle，证明两处均有合法第一步、连续局部退出以及局部heat距离下降；未向任何生产决策提供真值地图。无法根据已保存CSV证明当时UAV能观察并选择这些路径。
当前IdealSensor向adapter提供visible_cells/position/heading/velocity/nearby_robots，未传occupancy标签；_visible_cells不剔除占据的目标格，LOS仅检查目标格之前的遮挡（sensor_models.py:33–92,135–143）。adapter._update_belief把所有传入cells写成FREE（policy_adapter.py:375–395）。这是接口事实；缺历史快照，不能证明它具体污染了哪条历史边，也不能把FREE直接当作可靠footprint通行证明。

## 最小修复设计：唯一推荐方向

extended-only Search override：真实位置接入 + 拒绝反馈恢复。位置在Search override的命令出口；保留runtime模型、radius=0.2和SafetyShield。
1. 向extended命令选择明确提供已经实际发生的运动拒绝反馈：robot_id、实际位置、尝试位移和reason。这只暴露本机接触证据，不查询任意真实地图点。
2. 检查真实位置→候选前缀，以及必要的图边，依据可观测且occupancy语义可靠的局部数据；拒绝事件只否定已尝试transition，不伪造整片障碍地图。
3. 复用已有图候选/BFS和MARVEL base action；受阻hop不再无条件盖掉探索动作。有可观测且合法前缀时绕行重接；证据不足时使用经过验证的探索/感知动作。现有base action本来是否合法尚未知，不能保证简单取消override就一定恢复。离线A*不进入生产，也不扫描全图swept edges。
4. 每mission根据实际移动量推进/退出recovery；允许短期绕行，但分别评估解除停车、重接图与Search进度。UAV0到原hop会更远，不能把能动等同任务进度成功。
不建议重改坐标窗口映射、heading控制或通用运动模型；当前override直接读取图coords，没有窗口decode错配证据。也不是局部几何无解。

## 仍需采集的最小状态

各机override前后、首拒绝tick、下一mission：full-precision position/heading/velocity；公开heat anchor；base action/hop/final action；current_index、对应局部图coords/邻接和候选ID；belief crop/frame、visible_cells及占据语义来源；实际拒绝proposal/reason；邻近UAV position/intended targets与shield结果。
物理阻塞已确认；上述采集仅用于认证可观测恢复。本轮没有启动新仿真或额外采集，不需重做450step。

## 产物与复现

以final/为准。保留初版及verified/中间证据：初版误将接近heat2m作为hop搜索停止条件，verified/改为真正到达hop；final/对齐extended连续heading。verify_results.py独立断言确实到原hop。
diagnose_final.py保存独立诊断；final/results.json含完整tick/path/source hash；final/collision_replay.csv有40个还原tick；final/recovery_actions.csv有四条路径共11个命令；final/verification.json保存独立验证；final/local_geometry.png为真值oracle图。
已运行且退出码0：

~~~bash
<local-home>/miniconda3/envs/marvel/bin/python -B artifacts/offline_search_recovery_20261009/diagnose_final.py
<local-home>/miniconda3/envs/marvel/bin/python -B artifacts/offline_search_recovery_20261009/verify_results.py
~~~

脚本拒绝覆盖已存在输出；再次运行请先将OUT指向新独立目录。
new_simulation_runs=0；training=0；production_changes=0；commits=0；pushed=NO。最后Git状态见git_status_after.txt。

OFFLINE_SEARCH_RECOVERY_FEASIBILITY = INCONCLUSIVE

物理恢复已证明；可观测恢复尚不能认证，oracle路径不可直接部署。
