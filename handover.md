# 项目交接文档 — 多人实时表情识别基础工程

版本：1.0  
作者：Tizzyj（会话整理）  
用途：将当前对话与工程状态交接给新的高效 AI 模型/代理，便于继续代码开发与实验执行。

---

## 一、项目简述（一句话）
基于 MediaPipe FaceMesh 的多人实��表情识别系统骨架：检测与对齐 → 跟踪（IOU/ByteTrack）→ 部署端模型（MobileNetV3 占位）推理 → track 层滑窗融合；后续任务包含 ByteTrack 集成、知识蒸馏、量化与导出（ONNX/TensorRT/TFLite）。

---

## 二、当前目标（高层）
- 在本地跑通并稳定运行实时 demo（摄像头输入），输出 track id、label、置信度并记录子模块耗时。
- 为后续集成 ByteTrack、训练/蒸馏、量化导出等工作提供清晰接口与基础工程。

---

## 三、当前代码结构（已包含文件）
- `main.py`：程序入口，摄像头/视频读入，调用检测、跟踪、推理、融合并可视化，统计耗时。
- `mediapipe_face.py`：MediaPipe Face Detection + FaceMesh（detect(frame) → 列表 of faces）
- `simple_tracker.py`：IOUTracker（占位实现），接口 `update(detections)` → list of `{id, box}`。
- `model.py`：`FERModel`（MobileNetV3 占位），接口 `load_checkpoint(path)`；`predict_logits(crop)` → numpy logits。
- `fusion.py`：`TrackFusion`，窗口缓存 logits 并做 softmax 平均输出标签/置信度。
- `utils.py`：绘图函数 `draw_overlay(frame, results)`。
- `requirements.txt`：依赖清单（mediapipe, opencv-python, torch, torchvision, timm, numpy, scipy）。
- `.vscode/launch.json`：VS Code 调试配置。
- `README.md`：运行与环境说明（简要）。

---

## 四、模块接口约定（便于替换/扩展）
- `mediapipe_face.detect(frame)`  
  - 输入：BGR 图像（numpy array）  
  - 输出：列表，每项 dict:
    - `box`: [x1,y1,x2,y2]（像素坐标）
    - `landmarks`: （可返回 468 点或 None）
    - `crop`: 裁剪或对齐后的人脸图（BGR numpy）
- `IOUTracker.update(detections)`  
  - 输入：detections 列表，每项 `[x1,y1,x2,y2]`  
  - 输出：列表，每项 `{'id': int, 'box': [x1,y1,x2,y2]}`
- `FERModel.predict_logits(crop)`  
  - 输入：crop BGR numpy 图像  
  - 输出：一维 numpy array（长度 = num_classes），表示 logits
- `TrackFusion`  
  - `push(tid, logits)`：加入 buffer  
  - `has_result(tid)`：bool  
  - `get_result(tid)` → `(label, conf)`

---

## 五、如何复现（快速步骤）
1. 在项目根目录创建并激活虚拟环境：  
   - `python -m venv .venv`  
   - Windows: `.venv\Scripts\activate`  
   - macOS/Linux: `source .venv/bin/activate`
2. 安装依赖：`pip install -r requirements.txt`  
   - 若使用 GPU，请按 PyTorch 官方安装带 CUDA 的包（示例见 README）。
3. 运行 demo：`python main.py --device cuda`（若无 GPU，用 `--device cpu`）
4. 退出 demo：按 `q`。

---

## 六、已实现 / 限制（当前状态）
已实现：
- MediaPipe 检测与简单裁剪；
- IOU 基础跟踪（占位）；
- MobileNetV3 占位推理（未经训练分类头，仅用于接口验证）；
- Track 层滑窗融合（窗口平均 softmax）；
- 实时可视化与耗时统计（detect/track/infer/total）。

限制 / 待完善：
- 跟踪为占位 IOUTracker，尚未集成 ByteTrack 或 appearance re-id；
- 检测裁剪未做 FaceMesh 仿射对齐（仅用 bbox 裁剪）；
- 未含训练脚本（teacher/student/KD/feature distill）；
- 未含量化（QAT/PTQ）或导出脚本（ONNX/TensorRT/TFLite）；
- 数据加载器与大规模训练流程未实现。

---

## 七、优先级待办（P0 → P1 → P2）
每项附带验收标准，便于自动判定完成度。

P0（必须优先）
1. 跑通并稳定 main.py（demo）  
   - 验收：在本地 5 分钟稳定运行，无崩溃，并产出子模块耗时统计（detect/track/infer/total）。
2. 将 `mediapipe_face.detect` 扩展返回 landmarks 并实现仿射对齐（输出标准尺寸，如 128×128）  
   - 验收：对若干图片可视化对齐效果；`crop` 为对齐后图像。
3. 用 ByteTrack 替换或接入 `simple_tracker`（适配器层）  
   - 验收：多人视频中跟踪 ID 更稳定（主观或统计说明）；`main.py` 接口兼容。

P1（功能完善）
4. 批量化推理（按帧把多 track crop 批量预测）  
   - 验收：在 RTX3060 上 inference 平均延迟下降（提供前后对比）。
5. 实现训练脚本（`train_teacher.py`、`train_student_kd.py`）支持至少 RAF-DB 数据加载与 logit KD  
   - 验收：能短时跑通训练并生成 checkpoint 与 val macro F1。
6. 改进融合为置信度加权滑窗  
   - 验收：融合降低跟踪内抖动率。

P2（部署/论文需求）
7. 量化与导出（QAT / ONNX / TensorRT / TFLite）  
   - 验收：导出并在目标平台推理，记录量化前后精度与延迟差异。
8. 消融实验（分辨率、剪枝、KD 超参）并输出对比表格与图。

---

## 八、接手给新模型/代理的可执行指令（直接粘贴）
建议直接发送以下三条任务指令给新代理，按顺序执行：

任务 A（立刻执行 — 跑通 demo 并记录）：
- 在项目根目录：
  1. 创建并激活虚拟环境，安装依赖；
  2. 运行 `python main.py --device cpu`（若可用 GPU，使用 `--device cuda`）；
  3. 截图 demo 运行画面并保存为 `artifacts/demo_screenshot.png`；
  4. 记录子模块耗时并写入 `report.md`（包含命令、环境、错误日志与修复说明）。
- 验收：`report.md` 存在且说明 demo 是否成功；如有代码修复，提交修改说明。

任务 B（替换 tracker 为 ByteTrack）：
- 步骤：
  1. clone ByteTrack；实现 `byetrack_adapter.py`（将 MediaPipe 检测转换成 ByteTrack 输入）；
  2. 在 `main.py` 中替换 `IOUTracker` 为 ByteTrack 适配器（保持输出格式 list of `{id, box}`）；
  3. 在多人视频上测试并记录跟踪稳定性（���观描述或统计 ID switch）。
- 验收：跟踪在多人视频中表现合理；提交修改与测试报告。

任务 C（实现训练与 KD 验证）：
- 实现 `train_teacher.py` 与 `train_student_kd.py`（支持 RAF-DB 或小样本集），包含 logit KD 基线（T=3）与简单数据增强。  
- 验收：能在小规模数据上跑通 3–5 epochs 并在 `report.md` 中记录验证 macro F1 与训练日志。

---

## 九、交付格式（每项任务完成需提交）
- 代码改动：Git patch 或 commit（若无法 push，请上传修改后的文件列表与补丁）。  
- 文档：`report.md` 包含运行命令、环境（Python、依赖）、seed、耗时、指标、截图/视频链接。  
- 运行说明：若新增依赖或运行步骤，补充 `README.md`。  
- 若做实验：提供 CSV/表格记录每次试验配置与结果（便于后续撰写论文）。

---

## 十、资源链接（参考）
- ByteTrack: https://github.com/ifzhang/ByteTrack  
- DeepSORT: https://github.com/nwojke/deep_sort  
- MediaPipe face: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector  
- 示例 demo: https://github.com/GauravPandit27/Real-Time-Facial-Expression-Detection-using-Mediapipe-and-OpenCV

---

## 十一、沟通约束（给接手代理）
- 所有代码修改保留原文件备份（例如 `file.py.bak`）。  
- 若需大文件或 GPU 训练资源，请先在 `report.md` 说明所需资源与估时，等待授权。  
- 报告必包含可复现的最小命令（一行）以重现结果。  
- 任何实验结果需说明重复次数与随机种子。

---

## 十二、下一步建议（由代理优先执行）
1. 执行任务 A（跑通 demo 并提交 `report.md`）。  
2. 执行任务 B（集成 ByteTrack），优先解决多人跟踪稳定性问题。  
3. 实现批量化推理与初步训练脚本（小样例验证 KD 流程）。

---

### 附：快速执行命令（可直接复制）
- 创建环境并安装：  
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- 运行 demo（CPU）：  
  ```bash
  python main.py --device cpu
  ```
- 运行 demo（GPU）：  
  ```bash
  python main.py --device cuda
  ```

---

文档结束。若需要我把该 Markdown 文件保存为压缩包或生成“一键创建工程文件”的 shell 脚本，请回复“生成脚本”或“打包 ZIP”。