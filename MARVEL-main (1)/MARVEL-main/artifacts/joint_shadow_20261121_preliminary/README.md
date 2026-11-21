# 初步只读证据，非最终立即执行判据

公开状态捕获：原型 `ad864dd`。保存输入的只读重枚举：`5c3d5a6`。
这23个边界没有记录当时的MARVEL base action，首步dry-run以own-pose hold作fallback。
因此其中4个跨角色prefix事件只证明条件性的执行原语支持，不能按最终严格合同标为
“与实际executor输入一致的立即执行机会”。旧JSON中的executable字段须按此前提解释。

最终判据已在`a98b9a3`收紧：缺少base action时immediately_executable=False；
真正补录base action的四域短采样和被动对照见相邻`joint_shadow_20261121`。
两套状态有重叠，不能合并成独立样本或性能实验。生产源码/权重均未变化。

若需逐字节复现本目录旧派生字段，使用本分支`5c3d5a6`的shadow源码；当前版本
会有意拒绝把缺少真实fallback的prefix称为立即执行。原公共输入、原始GPPO动作、
wall时间和短窗口截断记录保留，不覆盖历史证据。

大型events文件已无损gzip；使用旧5c3d5a6读取前先在独立临时目录解压。packaging.json记录原始与容器SHA256。
