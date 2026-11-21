# 联合候选只读原型：复现入口

工作目录：`/workspace/2026PD_MARVEL/MARVEL-main (1)/MARVEL-main`。交接基底 `5692ca4`；最终严格原型 `a98b9a3`。生产源码、Rule、模型权重未修改。主证据 `artifacts/joint_shadow_20261121`，初步证据单独保存在 `artifacts/joint_shadow_20261121_preliminary`。

## CPU 环境与校验

实际 Python3.12.14、torch2.5.1+cpu、numpy1.26.4，无CUDA。可用Python3.11/3.12新venv；Torch单独使用CPU索引。

```bash
cd '/workspace/2026PD_MARVEL/MARVEL-main (1)/MARVEL-main'
python -m venv /workspace/marvel-audit-venv
/workspace/marvel-audit-venv/bin/pip install -r handoff/cloud/requirements-joint-shadow.txt
/workspace/marvel-audit-venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
export XDG_CACHE_HOME=/tmp/marvel-cache
export MPLCONFIGDIR=/tmp/marvel-mpl
/workspace/marvel-audit-venv/bin/python -B scripts/verify_joint_shadow.py \
  --out /tmp/marvel-joint-verification.json
```

检查交接428源码/配置哈希、8模型SHA/strict CPU load/masked forward，运行91项相关测试（其中13项新shadow测试）。仅在pytest进程中把旧测试模块的历史绝对checkpoint路径映射到已校验的交接资产；原测试和生产源码不改。新增Shapely依赖供真实MARVEL rollout导入，模型smoke原依赖列表不含它。

## 无仿真的精确重放

每case的 `public_state_NNN.json.gz` 是白名单完整枚举输入，包括当时冻结MARVEL实际输出的base actions。无GPU、不加载策略权重、不读取真实地图，即可复现候选、mask、prefix证书、分类和metrics。输出目录必须不存在。

```bash
/workspace/marvel-audit-venv/bin/python -B scripts/audit_joint_candidates.py \
  --replay artifacts/joint_shadow_20261121/30_static_1280021 \
  --out /tmp/marvel-joint-replay-30s
```

其他case为 `30_dynamic_1280022`、`60_static_9164`、`60_dynamic_1981204`。`--replay`逐事件比较候选、潜在图、dormant目录、metrics和公开digest，不重新推断原GPPO动作；原动作来自真实执行时observer。

交付将大型JSON/原始CSV逐字节无损压缩为`.gz`，`packaging.json`同时记录压缩前后SHA256，避免巨大的逐行数据diff；重放入口兼容plain或gzip事件文件。

`events.json.gz`包含实际allocator图/选择、落实后所有权、原scheduler waypoint和最终adapter命令。`gppo/physical_observations.json.gz`记录正常物理推进返回的own-sensor姿态、heading和motion反馈，没有额外oracle读取。该目录内原benchmark结果包含离线oracle覆盖率summary，枚举器不读取它，也不把它作为在线特征。

## 实际最终短采样命令

```bash
/workspace/marvel-audit-venv/bin/python -B scripts/audit_joint_candidates.py \
  --out /tmp/marvel-joint-exact-context --steps 3 \
  --wall-seconds-per-case 180 --verify-passive
```

四域各3控制步；另对30静态同条件运行3步无shadow对照。两臂物理observations、配置、初态、分配决策、碰撞、进度字节一致；事件仅排除顶层wall-clock `timestamp`后比较全部语义字段。原始日志和比较proof保留。每边界在线检查两次枚举结果一致、真实相关状态/RNG相同；每case核验冻结模型tensor指纹不变。

墙时限制在控制周期结束检查，可能略超预算；只报告实际边界。本命令只作接口发现，不得解释为H300验收。已存证据可直接重放，不必再跑此采样。

## 初步证据与已纠正的局限

第一版 `ad864dd` 运行同四域 `--steps 6 --wall-seconds-per-case 180`，得到23边界（60静态5步，其他6步）；`5c3d5a6`只用保存公开状态重枚举，未重跑物理。它以own-pose hold作为dry-run fallback，缺少真实MARVEL base actions。因此4个跨角色prefix只能作条件性证据，不能使用最终严格“立即执行”标签。旧产物单独保留，并记录本次收紧；最终12边界是补录真实fallback后的主证据，两套不能合并计独立样本。

初始依赖探测缺Shapely；原型调试修正frozenset序列化与隔离runtime.frame字段。第一版被动比较因原始wall timestamp不同而退出非零，语义字段实际一致；最终工具显式排除此唯一字段，并额外比对逐tick物理观测。失败记录不用于收益或安全结论。

## 组件与边界

- `integrations/gppo/joint_shadow.py`：白名单投影、显式JSON、隔离原调度/掩码复用、原运动接口首步证书、分类/统计。
- `integrations/gppo/joint_shadow_recorder.py`：仅当前实例的observer，转发原assign一次，close恢复方法；捕获实际base动作，不修改执行输入。
- `scripts/audit_joint_candidates.py`：短CPU采集、精确重放、保存状态重枚举和被动对照。
- `scripts/verify_joint_shadow.py`：可移植checkpoint校验和回归测试。
- `platform_tests/test_joint_shadow.py`：13项信息流、资源、Safety、deadline/可达性、模型/时钟/物理状态不变性测试。

首步证书时域仅.1s：same recorded base action、own观测姿态、观测FREE五探针、公开warning及delayed peers最大速度球。缺真实base action时只能给conditional_prefix，不能标immediately_executable。UNKNOWN残差、radio预测、dormant目录、合法分配边均不代表完整路径或全窗安全。枚举不调用真实step，不推进时钟、不写真实任务、不采样、不改模型。
