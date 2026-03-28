import numpy as np
import torch
from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor

class TravelModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        self.tokenizer, self.model, self.image_processor = load_model(model_args)
        self.traj_model = load_traj_model(model_args)
        self.model.to(torch.bfloat16)
        self._force_vggt_fp32()
        self.traj_model.to(dtype=torch.bfloat16, device=self.model.device)
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

    def _force_vggt_fp32(self):
        candidate_modules = []
        if hasattr(self.model, "vggt_model"):
            candidate_modules.append(self.model.vggt_model)
        base_model = getattr(self.model, "base_model", None)
        if base_model is not None and hasattr(base_model, "model") and hasattr(base_model.model, "vggt_model"):
            candidate_modules.append(base_model.model.vggt_model)

        visited = set()
        for module in candidate_modules:
            if module is None or id(module) in visited:
                continue
            module.to(device=self.model.device, dtype=torch.float32)
            module.eval()
            visited.add(id(module))

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        inputs = []
        rot_to_targets = []
        
        for i in range(len(episodes)):
            input_item, rot_to_target = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None
            )
            inputs.append(input_item)
            rot_to_targets.append(rot_to_target)
        batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs)

        inputs_device = {k: v.to(self.model.device) for k, v in batch.items() 
            if 'prompts' not in k and 'images' not in k and 'historys' not in k}
        inputs_device['prompts'] = [item for item in batch['prompts']]
        inputs_device['images'] = [item.to(self.model.device) for item in batch['images']]
        # ==========================================
        # [新增] 放置 VGGT 图像并对齐精度 (bf16/fp16)
        # ==========================================
        if 'vggt_images' in batch:
            inputs_device['vggt_images'] = batch['vggt_images'].to(device=self.model.device, dtype=torch.float32)
        # ==========================================
        inputs_device['historys'] = [item.to(device=self.model.device, dtype=self.model.dtype) for item in batch['historys']]
        inputs_device['orientations'] = inputs_device['orientations'].to(dtype=self.model.dtype)
        inputs_device['return_waypoints'] = True
        inputs_device['use_cache'] = False
        
        return inputs_device, rot_to_targets

    def run_llm_model(self, inputs):
        waypoints_llm = self.model(**inputs).cpu().to(dtype=torch.float32).numpy()
        waypoints_llm = np.nan_to_num(waypoints_llm, nan=0.0, posinf=0.0, neginf=0.0)
        waypoints_llm_new = []
        for waypoint in waypoints_llm:
            direction = waypoint[:3]
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-6:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                direction_norm = 1.0
            distance = float(np.clip(waypoint[3], 0.0, 30.0))
            waypoint_new = direction / (1e-6 + direction_norm) * distance
            waypoints_llm_new.append(waypoint_new)
        return np.array(waypoints_llm_new)

    def run_traj_model(self, episodes, waypoints_llm_new, rot_to_targets):
        inputs = prepare_data_to_traj_model(episodes, waypoints_llm_new, self.image_processor, rot_to_targets)
        waypoints_traj = self.traj_model(inputs, None)
        refined_waypoints = waypoints_traj.cpu().to(dtype=torch.float32).numpy()
        refined_waypoints = transform_to_world(refined_waypoints, episodes)
        return refined_waypoints
    
    def eval(self):
        self.model.eval()
        self.traj_model.eval()
        
    def run(self, inputs, episodes, rot_to_targets):
        waypoints_llm_new = self.run_llm_model(inputs)
        refined_waypoints = self.run_traj_model(episodes, waypoints_llm_new, rot_to_targets)
        fixed_refined_waypoints = []
        invalid_traj_count = 0
        for i in range(len(refined_waypoints)):
            traj = np.asarray(refined_waypoints[i], dtype=np.float32)
            if traj.ndim != 2 or traj.shape[1] != 3 or (not np.isfinite(traj).all()):
                invalid_traj_count += 1
                ep = episodes[i]
                state = ep[-1]["sensors"]["state"]
                pos = np.asarray(state["position"], dtype=np.float32)
                rot = np.asarray(ep[-1]["sensors"]["imu"]["rotation"], dtype=np.float32)
                if rot.shape == (3, 3):
                    forward = rot[:, 0]
                else:
                    forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                forward = forward / (np.linalg.norm(forward) + 1e-6)
                traj = np.stack([pos + forward * float(step + 1) for step in range(7)], axis=0)
            fixed_refined_waypoints.append(traj)
        refined_waypoints = fixed_refined_waypoints
        return refined_waypoints
    
    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(episodes[i], object_infos[i])
            prediction_dones.append(prediction_done)
        return prediction_dones
        

    
