# 联合候选第一阶段交付发布状态

本地仓库 `/workspace/2026PD_MARVEL`，分支 `codex/joint-gppo-shadow-20261121`，父基底为指定交接提交 `5692ca4fe5c56f398707deea87c1efc844297b71`。所有源码、报告与证据已提交，生产原文件/模型没有修改。

远端尚未发布：

- `git push -u origin codex/joint-gppo-shadow-20261121` 返回GitHub403，当前Git身份无此仓库推送权限。
- 已连接GitHub工具的仓库metadata显示push权限，但实际`create_branch`请求也返回403 `Resource not accessible by integration`。
- 未创建远端新分支或PR；未合并、未更新交接分支、未改任何远端历史。

已生成包含完整本地新增提交的可验证bundle：`/workspace/joint-gppo-shadow-20261121.bundle`。它只包含交接提交之后的新对象，导入方需已有该交接基底。源代码提交`ad864dd`、`5c3d5a6`、`a98b9a3`及最终报告提交均随bundle保留，便于核对capture provenance。

在已有交接仓库的本机、使用有权限的凭据：

```bash
git bundle verify /workspace/joint-gppo-shadow-20261121.bundle
git fetch /workspace/joint-gppo-shadow-20261121.bundle refs/heads/codex/joint-gppo-shadow-20261121
git push origin FETCH_HEAD:refs/heads/codex/joint-gppo-shadow-20261121
```

这些命令不更新交接分支，也不使用force。无需重新运行采样或训练。
