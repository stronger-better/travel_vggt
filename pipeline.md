```mermaid
graph TD
    %% 数据输入层
    subgraph Data_Input ["1. 数据输入 (Dataloader)"]
        A1["无人机多视角图像帧<br/>Historical & Current Views"]
        A2["文本指令与历史轨迹<br/>Text Prompts & History"]
        A3["真实未来航点序列<br/>Ground Truth Waypoints"]
    end

    %% 预处理层
    subgraph Preprocessing ["2. 预处理 (Transforms)"]
        B1["CLIP 图像处理<br/>Standard Normalization"]
        B2["VGGT 图像处理<br/>[0, 1] Normalization"]
        B3["文本 Tokenizer<br/>Encode Input IDs"]
    end

    A1 --> B1
    A1 --> B2
    A2 --> B3

    %% 视觉特征提取层
    subgraph Vision_Extraction ["3. 视觉特征提取 (Dual Vision Towers)"]
        C1["LLaMA-VID 视觉塔<br/>提取 2D 语义特征"]
        C2["VGGT Aggregator<br/>提取高维 3D Latent 几何特征"]
        C3["VGGT Projector<br/>下采样与隐层维度对齐"]
    end

    B1 --> C1
    B2 --> C2
    C2 --> C3

    %% 多模态融合层
    subgraph Fusion ["4. 多模态 Token 融合 (llamavid_arch.py)"]
        D1["定位占位符<br/>image, his, wp"]
        D2["拼接重组 Input Embeddings<br/>Text + 2D Tokens + 3D Tokens"]
    end

    B3 --> D1
    C1 --> D2
    C3 --> D2
    D1 --> D2

    %% 语言模型核心
    subgraph LLM ["5. 大语言模型推理 (llava_llama_uav.py)"]
        E1["LLaMA / Qwen Backbone<br/>自回归因果注意力"]
        E2["输出全局 Hidden States"]
    end

    D2 --> E1
    E1 --> E2

    %% 动作预测与损失计算
    subgraph Action_Loss ["6. 动作预测与损失计算 (Action Head)"]
        F1["提取特定位置状态<br/>Extract labels == WAYPOINT_TOKEN"]
        F2["无人机动作专家 (MLP)<br/>waypoints_fc"]
        F3["预测 3D 相对航点<br/>dx, dy, dz, yaw"]
        F4(("Loss Calculation"))
        F5["L1 距离损失<br/>Waypoint Loss"]
        F6["余弦方向损失<br/>Angle Loss"]
    end

    E2 --> F1
    F1 --> F2
    F2 --> F3
    F3 --> F4
    A3 --> F4
    F4 --> F5
    F4 --> F6

    %% 反向传播
    F5 -.->|Backpropagation| Model_Weights[("更新模型权重")]
    F6 -.->|Backpropagation| Model_Weights