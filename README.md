# 多人实时表情识别基础工程（骨架）

说明
- 最小可运行 demo：MediaPipe FaceMesh -> 简单 IOU 跟踪 -> MobileNetV3 占位推理 -> track 层滑窗融合 -> 可视化结果
- 目标：作为后续接入 ByteTrack、知识蒸馏、QAT 的工程起点

依赖（参见 requirements.txt）
- Python 3.8–3.10, mediapipe, opencv-python, torch, torchvision, timm, numpy

快速运行（推荐在 conda/venv 中）
1. 克隆或把本工程放到本地文件夹
2. 创建虚拟环境并激活：
   - python -m venv .venv
   - Windows: .venv\\Scripts\\activate
   - Linux/macOS: source .venv/bin/activate
3. 安装依赖：
   - pip install -r requirements.txt
   - 注意：若使用 GPU，请按 PyTorch 官方指引安装匹配 CUDA 的 torch（例如 pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116）
4. 运行 demo：
   - python main.py --device cuda   # 在有 GPU 时用 cuda，否则用 cpu

VS Code 导入（简要见 docs/ 下的说明或 README 下方）