"""
模型管理器
Model Manager for PyTorch Models
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import cv2
from ultralytics import YOLO
import torchvision.transforms as transforms
import threading
import gc
from functools import lru_cache
import time

from .data_models import Detection, BoundingBox, ModelConfig
from .exceptions import ModelLoadError, InferenceError, ConfigurationError
from .config import config_manager


class ModelManager:
    """深度学习模型管理器 - 性能优化版本"""
    
    def __init__(self):
        self.vehicle_models: Dict[str, Any] = {}
        self.water_models: Dict[str, Any] = {}
        self.current_vehicle_model: Optional[str] = None
        self.current_water_model: Optional[str] = None
        self.device = self._get_device()
        
        # 性能优化设置
        self._model_cache = {}
        self._inference_lock = threading.Lock()
        self._warmup_done = False
        self._batch_size = 1
        
        # 优化的预处理变换 - 使用更快的插值方法
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 设置PyTorch性能优化
        self._setup_torch_optimizations()
    
    def _get_device(self) -> torch.device:
        """获取可用的计算设备"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用GPU: {torch.cuda.get_device_name()}")
            # 设置GPU内存优化
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
        else:
            device = torch.device("cpu")
            print("使用CPU进行推理")
            # CPU优化设置
            torch.set_num_threads(min(4, torch.get_num_threads()))  # 限制CPU线程数
        return device
    
    def _setup_torch_optimizations(self):
        """设置PyTorch性能优化"""
        # 启用cudnn基准测试以优化卷积操作
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # 设置内存分配策略
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    def load_models(self) -> bool:
        """加载所有可用的模型"""
        success = True
        
        # 加载车辆检测模型
        for model_config in config_manager.config.vehicle_models:
            try:
                print(f"⏳ 正在加载车辆检测模型: {model_config.name}...")
                self._load_vehicle_model(model_config)
                print(f"✅ 车辆检测模型加载成功: {model_config.name}")
            except Exception as e:
                print(f"❌ 车辆检测模型加载失败: {model_config.name} - {e}")
                success = False
        
        # 加载水面分割模型
        for model_config in config_manager.config.water_models:
            try:
                print(f"⏳ 正在加载水面分割模型: {model_config.name}...")
                self._load_water_model(model_config)
                print(f"✅ 水面分割模型加载成功: {model_config.name}")
            except Exception as e:
                print(f"❌ 水面分割模型加载失败: {model_config.name} - {e}")
                success = False
        
        # 设置默认模型
        if self.vehicle_models:
            self.current_vehicle_model = list(self.vehicle_models.keys())[0]
        if self.water_models:
            self.current_water_model = list(self.water_models.keys())[0]
        
        # 执行模型预热
        if success and self.current_vehicle_model and self.current_water_model:
            self._warmup_models()
        
        return success
    
    def _warmup_models(self):
        """预热模型以提高首次推理速度"""
        if self._warmup_done:
            return
            
        try:
            print("正在预热模型...")
            start_time = time.time()
            
            # 创建虚拟输入进行预热
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 预热车辆检测模型
            if self.current_vehicle_model:
                try:
                    self.predict_vehicles(dummy_image)
                except:
                    pass  # 忽略预热时的错误
            
            # 预热水面分割模型
            if self.current_water_model:
                try:
                    self.predict_water(dummy_image)
                except:
                    pass  # 忽略预热时的错误
            
            self._warmup_done = True
            warmup_time = time.time() - start_time
            print(f"模型预热完成，耗时: {warmup_time:.2f}s")
            
        except Exception as e:
            print(f"模型预热失败: {e}")
    
    @lru_cache(maxsize=32)
    def _get_cached_preprocessing(self, image_shape: Tuple[int, int], model_type: str) -> Any:
        """缓存预处理参数"""
        # 这里可以缓存一些预处理相关的计算结果
        return {
            'input_shape': image_shape,
            'model_type': model_type,
            'cached_at': time.time()
        }
    
    def _load_vehicle_model(self, model_config: ModelConfig):
        """加载车辆检测模型 - 性能优化版本"""
        if not os.path.exists(model_config.file_path):
            raise ModelLoadError(model_config.file_path, "模型文件不存在")
        
        try:
            if "yolov11" in model_config.name.lower():
                # 使用Ultralytics加载YOLOv11模型
                try:
                    model = YOLO(model_config.file_path)
                except Exception as e:
                    if "weights_only" in str(e):
                        # 临时设置环境变量以允许不安全的加载
                        old_weights_only = os.environ.get('TORCH_WEIGHTS_ONLY', None)
                        os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
                        try:
                            model = YOLO(model_config.file_path)
                        finally:
                            if old_weights_only is not None:
                                os.environ['TORCH_WEIGHTS_ONLY'] = old_weights_only
                            else:
                                os.environ.pop('TORCH_WEIGHTS_ONLY', None)
                    else:
                        raise e
                
                model.to(self.device)
                
                # 性能优化设置
                if hasattr(model.model, 'half') and self.device.type == 'cuda':
                    model.model.half()  # 使用半精度推理
                    
            elif "rt-detr" in model_config.name.lower() or "rtdetr" in model_config.name.lower():
                # RT-DETR模型使用YOLO接口加载
                try:
                    print(f"正在使用YOLO接口加载RT-DETR: {model_config.file_path}")
                    model = YOLO(model_config.file_path)
                    print(f"YOLO加载完成，模型类型: {type(model)}")
                    model.to(self.device)
                    
                    # 性能优化设置
                    if hasattr(model.model, 'half') and self.device.type == 'cuda':
                        model.model.half()  # 使用半精度推理
                    
                    print(f"✅ RT-DETR模型通过YOLO接口加载成功，最终类型: {type(model)}")
                    
                except Exception as e:
                    raise ModelLoadError(model_config.file_path, f"RT-DETR加载失败: {e}")
                
            else:
                # 通用PyTorch模型加载，处理weights_only问题
                try:
                    model = torch.load(model_config.file_path, map_location=self.device, weights_only=False)
                except TypeError:
                    # 旧版本PyTorch不支持weights_only参数
                    model = torch.load(model_config.file_path, map_location=self.device)
                
                if hasattr(model, 'eval'):
                    model.eval()
                
                # 启用半精度推理（如果支持）
                if self.device.type == 'cuda' and hasattr(model, 'half'):
                    model.half()
            
            # 编译模型以提高推理速度（PyTorch 2.0+）
            # 只对可调用的PyTorch模型进行编译，跳过YOLO和RT-DETR模型
            if (hasattr(torch, 'compile') and 
                not isinstance(model, YOLO) and 
                "ultralytics" not in str(type(model)).lower() and
                hasattr(model, '__call__') and 
                hasattr(model, 'parameters') and
                callable(model) and
                "rt-detr" not in model_config.name.lower() and
                "rtdetr" not in model_config.name.lower()):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    print(f"✅ 车辆检测模型编译成功: {model_config.name}")
                except Exception as e:
                    print(f"⚠️ 车辆检测模型编译失败，使用原始模型: {e}")
            
            print(f"存储模型 {model_config.name}，类型: {type(model)}")
            self.vehicle_models[model_config.name] = model
            
        except Exception as e:
            raise ModelLoadError(model_config.file_path, str(e))
    
    def _load_water_model(self, model_config: ModelConfig):
        """加载水面分割模型 - 性能优化版本"""
        if not os.path.exists(model_config.file_path):
            raise ModelLoadError(model_config.file_path, "模型文件不存在")
        
        try:
            if "yolov11" in model_config.name.lower():
                # YOLOv11分割模型，处理PyTorch 2.6兼容性
                try:
                    model = YOLO(model_config.file_path)
                except Exception as e:
                    if "weights_only" in str(e):
                        # 临时设置环境变量以允许不安全的加载
                        old_weights_only = os.environ.get('TORCH_WEIGHTS_ONLY', None)
                        os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
                        try:
                            model = YOLO(model_config.file_path)
                        finally:
                            if old_weights_only is not None:
                                os.environ['TORCH_WEIGHTS_ONLY'] = old_weights_only
                            else:
                                os.environ.pop('TORCH_WEIGHTS_ONLY', None)
                    else:
                        raise e
                
                model.to(self.device)
                
                # 性能优化设置
                if hasattr(model.model, 'half') and self.device.type == 'cuda':
                    model.model.half()  # 使用半精度推理
                    
            elif "deeplabv3" in model_config.name.lower():
                # DeepLabV3分割模型，处理weights_only问题
                try:
                    loaded_data = torch.load(model_config.file_path, map_location=self.device, weights_only=False)
                except TypeError:
                    # 旧版本PyTorch不支持weights_only参数
                    loaded_data = torch.load(model_config.file_path, map_location=self.device)
                
                # 检查加载的数据类型
                if isinstance(loaded_data, (dict, torch.nn.modules.container.OrderedDict)):
                    # 如果是状态字典，创建DeepLabV3模型并加载权重
                    print(f"正在为 {model_config.name} 创建DeepLabV3模型架构...")
                    model = self._create_deeplabv3_model()
                    
                    # 加载状态字典
                    try:
                        model.load_state_dict(loaded_data, strict=False)
                        print(f"✅ DeepLabV3权重加载成功")
                    except Exception as e:
                        print(f"警告: DeepLabV3权重加载部分失败: {e}")
                    
                    model.to(self.device)
                    model.eval()
                    
                    # 启用半精度推理（如果支持）
                    if self.device.type == 'cuda':
                        model.half()
                else:
                    model = loaded_data
                    if hasattr(model, 'eval'):
                        model.eval()
                    
                    # 启用半精度推理（如果支持）
                    if self.device.type == 'cuda' and hasattr(model, 'half'):
                        model.half()
                    
            else:
                # 通用PyTorch模型加载，处理weights_only问题
                try:
                    model = torch.load(model_config.file_path, map_location=self.device, weights_only=False)
                except TypeError:
                    # 旧版本PyTorch不支持weights_only参数
                    model = torch.load(model_config.file_path, map_location=self.device)
                
                if hasattr(model, 'eval'):
                    model.eval()
                
                # 启用半精度推理（如果支持）
                if self.device.type == 'cuda' and hasattr(model, 'half'):
                    model.half()
            
            # 编译模型以提高推理速度（PyTorch 2.0+）
            # 只对可调用的PyTorch模型进行编译
            if (hasattr(torch, 'compile') and 
                not isinstance(model, YOLO) and 
                hasattr(model, '__call__') and 
                hasattr(model, 'parameters') and
                callable(model)):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    print(f"✅ 水面分割模型编译成功: {model_config.name}")
                except Exception as e:
                    print(f"⚠️ 水面分割模型编译失败，使用原始模型: {e}")
            
            self.water_models[model_config.name] = model
            
        except Exception as e:
            raise ModelLoadError(model_config.file_path, str(e))
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用模型列表"""
        return {
            'vehicle_models': list(self.vehicle_models.keys()),
            'water_models': list(self.water_models.keys())
        }
    
    def set_active_models(self, vehicle_model: str, water_model: str) -> bool:
        """设置当前活跃的模型"""
        success = True
        
        if vehicle_model not in self.vehicle_models:
            print(f"警告: 车辆检测模型不存在: {vehicle_model}")
            success = False
        else:
            self.current_vehicle_model = vehicle_model
        
        if water_model not in self.water_models:
            print(f"警告: 水面分割模型不存在: {water_model}")
            success = False
        else:
            self.current_water_model = water_model
        
        return success
    
    def predict_vehicles(self, image: np.ndarray) -> List[Detection]:
        """使用当前车辆检测模型进行预测 - 性能优化版本"""
        if not self.current_vehicle_model:
            raise InferenceError("车辆检测", "未设置活跃的车辆检测模型")
        
        if self.current_vehicle_model not in self.vehicle_models:
            raise InferenceError("车辆检测", f"模型不存在: {self.current_vehicle_model}")
        
        # 使用线程锁确保推理安全
        with self._inference_lock:
            try:
                model = self.vehicle_models[self.current_vehicle_model]
                
                # 获取配置
                config = None
                for model_config in config_manager.config.vehicle_models:
                    if model_config.name == self.current_vehicle_model:
                        config = model_config
                        break
                
                if config is None:
                    raise InferenceError("车辆检测", f"找不到模型配置: {self.current_vehicle_model}")
                
                # 内存优化：在推理前清理缓存
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # 根据模型类型进行推理
                if isinstance(model, YOLO) or "ultralytics" in str(type(model)).lower():
                    return self._predict_with_yolo_optimized(model, image, config)
                else:
                    return self._predict_with_pytorch_model_optimized(model, image, config, 'detection')
                    
            except Exception as e:
                raise InferenceError(self.current_vehicle_model, str(e))
    
    def predict_water(self, image: np.ndarray) -> np.ndarray:
        """使用当前水面分割模型进行预测 - 性能优化版本"""
        if not self.current_water_model:
            raise InferenceError("水面分割", "未设置活跃的水面分割模型")
        
        if self.current_water_model not in self.water_models:
            raise InferenceError("水面分割", f"模型不存在: {self.current_water_model}")
        
        # 使用线程锁确保推理安全
        with self._inference_lock:
            try:
                model = self.water_models[self.current_water_model]
                
                # 获取配置
                config = None
                for model_config in config_manager.config.water_models:
                    if model_config.name == self.current_water_model:
                        config = model_config
                        break
                
                if config is None:
                    raise InferenceError("水面分割", f"找不到模型配置: {self.current_water_model}")
                
                # 内存优化：在推理前清理缓存
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # 根据模型类型进行推理
                if isinstance(model, YOLO) or "ultralytics" in str(type(model)).lower():
                    return self._predict_water_with_yolo_optimized(model, image, config)
                else:
                    return self._predict_water_with_pytorch_model_optimized(model, image, config)
                    
            except Exception as e:
                raise InferenceError(self.current_water_model, str(e))
    
    def _predict_with_yolo_optimized(self, model: YOLO, image: np.ndarray, config: ModelConfig) -> List[Detection]:
        """使用YOLO模型进行车辆检测 - 性能优化版本"""
        
        # 🔥 回退修复：让YOLO使用原生预处理
        # 避免双重预处理问题，让Ultralytics自己处理预处理
        
        # 优化的YOLO推理设置
        with torch.no_grad():  # 禁用梯度计算
            results = model(
                image,  # 直接使用原始图像，让YOLO自己预处理
                conf=config.confidence_threshold, 
                iou=config.nms_threshold,
                verbose=False,  # 禁用详细输出
                device=self.device,
                half=self.device.type == 'cuda'  # 使用半精度（如果是GPU）
            )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                # 批量处理以提高效率
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # 向量化处理
                # 定义淹没等级映射（基于你的训练数据）
                flood_level_mapping = {
                    0: 'cc',  # 车窗及以上 (ROOF_LEVEL)
                    1: 'cm',  # 车轮顶部至车窗下沿 (WINDOW_LEVEL)  
                    2: 'lt'   # 车轮顶部及以下 (WHEEL_LEVEL)
                }
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    bbox = BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(conf)
                    )
                    
                    # 使用淹没等级作为类别名称
                    flood_level_code = flood_level_mapping.get(int(cls_id), f'unknown_{cls_id}')
                    
                    detection = Detection(
                        bbox=bbox,
                        class_id=int(cls_id),
                        class_name=f"vehicle_{flood_level_code}"  # 标识为车辆+淹没等级
                    )
                    detections.append(detection)
        
        return detections
    
    def _predict_with_yolo(self, model: YOLO, image: np.ndarray, config: ModelConfig) -> List[Detection]:
        """使用YOLO模型进行车辆检测 - 兼容性方法"""
        return self._predict_with_yolo_optimized(model, image, config)
    
    def _predict_with_pytorch_model_optimized(self, model, image: np.ndarray, config: ModelConfig, task: str) -> List[Detection]:
        """使用通用PyTorch模型进行检测 - 性能优化版本"""
        
        # 检查模型是否可调用
        if isinstance(model, dict):
            # 如果是字典格式，尝试提取模型
            if 'model' in model:
                actual_model = model['model']
            elif 'state_dict' in model:
                raise InferenceError("模型推理", "检测到state_dict格式，需要先加载到模型架构中")
            else:
                raise InferenceError("模型推理", "不支持的模型格式：字典类型但无法找到模型对象")
        else:
            actual_model = model
        
        # 确保模型可调用
        if not callable(actual_model):
            raise InferenceError("模型推理", f"模型不可调用，类型: {type(actual_model)}")
        
        # 优化的预处理图像
        if len(image.shape) == 3:
            # 使用更快的预处理方法
            input_tensor = self._fast_preprocess(image, config.input_size).to(self.device, non_blocking=True)
        else:
            raise ValueError("输入图像格式错误")
        
        # 优化的模型推理
        try:
            # 确保输入数据类型正确
            if hasattr(actual_model, 'parameters'):
                # 获取模型的数据类型
                model_dtype = next(actual_model.parameters()).dtype
                input_tensor = input_tensor.to(dtype=model_dtype)
            
            if self.device.type == 'cuda':
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    outputs = actual_model(input_tensor)
            else:
                with torch.no_grad():
                    outputs = actual_model(input_tensor)
        except Exception as e:
            print(f"警告: {config.name} 模型推理失败: {e}，返回空检测结果")
            return []
        
        # 后处理（这里需要根据具体模型输出格式调整）
        detections = []
        # 注意：这里需要根据实际模型的输出格式来解析结果
        # 对于RT-DETR等检测模型，需要根据具体输出格式解析
        
        # 临时返回空列表，避免推理失败
        print(f"警告: {config.name} 模型推理暂未完全实现，返回空检测结果")
        
        return detections
    
    def _predict_with_pytorch_model(self, model, image: np.ndarray, config: ModelConfig, task: str) -> List[Detection]:
        """使用通用PyTorch模型进行检测 - 兼容性方法"""
        return self._predict_with_pytorch_model_optimized(model, image, config, task)
    
    def _predict_water_with_yolo_optimized(self, model: YOLO, image: np.ndarray, config: ModelConfig) -> np.ndarray:
        """使用YOLO分割模型进行水面分割 - 性能优化版本"""
        
        # 🔥 回退修复：让YOLO使用原生预处理
        # 避免双重预处理问题
        
        # 优化的YOLO分割推理
        with torch.no_grad():
            results = model(
                image,  # 直接使用原始图像
                conf=config.confidence_threshold,
                verbose=False,
                device=self.device,
                half=self.device.type == 'cuda'
            )
        
        # 创建空的掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for result in results:
            if result.masks is not None:
                # 获取分割掩码 - 批量处理
                masks = result.masks.data.cpu().numpy()
                
                # 优化的掩码处理
                for seg_mask in masks:
                    # 使用更快的插值方法调整掩码尺寸
                    if seg_mask.shape != image.shape[:2]:
                        resized_mask = cv2.resize(
                            seg_mask, 
                            (image.shape[1], image.shape[0]), 
                            interpolation=cv2.INTER_NEAREST  # 更快的插值方法
                        )
                    else:
                        resized_mask = seg_mask
                    
                    # 向量化操作
                    binary_mask = (resized_mask > 0.5).astype(np.uint8)
                    mask = np.maximum(mask, binary_mask)
        
        return mask
    
    def _predict_water_with_yolo(self, model: YOLO, image: np.ndarray, config: ModelConfig) -> np.ndarray:
        """使用YOLO分割模型进行水面分割 - 兼容性方法"""
        return self._predict_water_with_yolo_optimized(model, image, config)
    
    def _predict_water_with_pytorch_model_optimized(self, model, image: np.ndarray, config: ModelConfig) -> np.ndarray:
        """使用PyTorch分割模型进行水面分割 - 性能优化版本"""
        
        # 检查模型是否可调用
        if isinstance(model, (dict, torch.nn.modules.container.OrderedDict)):
            # 如果是字典或OrderedDict格式，尝试提取模型
            if 'model' in model:
                actual_model = model['model']
            elif 'state_dict' in model:
                raise InferenceError("模型推理", "检测到state_dict格式，需要先加载到模型架构中")
            else:
                # 对于DeepLabV3等模型，可能直接是OrderedDict状态字典
                print(f"警告: {config.name} 模型为状态字典格式，暂未完全实现，返回空掩码")
                return np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            actual_model = model
        
        # 确保模型可调用
        if not callable(actual_model):
            print(f"警告: {config.name} 模型不可调用，类型: {type(actual_model)}，返回空掩码")
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 优化的预处理图像
        if len(image.shape) == 3:
            input_tensor = self._fast_preprocess(image, config.input_size, 'deeplabv3').to(self.device, non_blocking=True)
        else:
            raise ValueError("输入图像格式错误")
        
        # 优化的模型推理
        try:
            if self.device.type == 'cuda':
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    outputs = actual_model(input_tensor)
            else:
                with torch.no_grad():
                    outputs = actual_model(input_tensor)
        except Exception as e:
            print(f"警告: {config.name} 模型推理失败: {e}，返回空掩码")
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 后处理
        if isinstance(outputs, dict) and 'out' in outputs:
            # DeepLabV3输出格式
            output = outputs['out']
        else:
            output = outputs
        
        # 处理单类别输出
        if output.shape[1] == 1:
            # 单类别输出，直接使用sigmoid激活
            pred_mask = torch.sigmoid(output).squeeze()
            
            # 直接在GPU上进行后处理（如果可能）
            if self.device.type == 'cuda':
                pred_mask = pred_mask.cpu()
            
            pred_mask = pred_mask.numpy()
            
            # 使用阈值进行二值化
            water_mask = (pred_mask > 0.5).astype(np.uint8)
        else:
            # 多类别输出，使用argmax
            pred_mask = torch.argmax(output, dim=1).squeeze()
            
            # 直接在GPU上进行后处理（如果可能）
            if self.device.type == 'cuda':
                pred_mask = pred_mask.cpu()
            
            pred_mask = pred_mask.numpy()
            
            # 二值化（假设水面类别为1）
            water_mask = (pred_mask == 1).astype(np.uint8)
        
        # 使用更快的插值方法调整尺寸到原图大小
        if water_mask.shape != image.shape[:2]:
            water_mask = cv2.resize(
                water_mask.astype(np.uint8), 
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        return water_mask
    
    def _predict_water_with_pytorch_model(self, model, image: np.ndarray, config: ModelConfig) -> np.ndarray:
        """使用PyTorch分割模型进行水面分割 - 兼容性方法"""
        return self._predict_water_with_pytorch_model_optimized(model, image, config)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        info = {
            'current_vehicle_model': self.current_vehicle_model,
            'current_water_model': self.current_water_model,
            'device': str(self.device),
            'available_models': self.get_available_models()
        }
        return info
    
    def unload_models(self):
        """卸载所有模型以释放内存"""
        self.vehicle_models.clear()
        self.water_models.clear()
        self.current_vehicle_model = None
        self.current_water_model = None
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 清理缓存
        self._model_cache.clear()
        self._get_cached_preprocessing.cache_clear()
        
        # 强制垃圾回收
        gc.collect()
        
        print("所有模型已卸载")
    
    def _fast_preprocess(self, image: np.ndarray, input_size: Tuple[int, int], model_type: str = 'deeplabv3') -> torch.Tensor:
        """快速预处理方法 - 根据模型类型使用正确的预处理"""
        # 使用OpenCV进行更快的预处理
        # 转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 快速缩放
        resized = cv2.resize(rgb_image, input_size, interpolation=cv2.INTER_LINEAR)
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 根据模型类型选择正确的归一化方式
        if model_type.lower() in ['deeplabv3', 'deeplab']:
            # DeepLabV3训练时使用的归一化：mean=[0,0,0], std=[1,1,1]
            # 相当于只做 /255.0，不做ImageNet标准化
            pass  # 已经做了 /255.0，不需要额外处理
        else:
            # 其他模型可能需要ImageNet标准化
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
        
        # 转换为张量
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        memory_info = {
            'models_loaded': len(self.vehicle_models) + len(self.water_models),
            'warmup_done': self._warmup_done
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**2,   # MB
                'gpu_memory_cached': torch.cuda.memory_cached() / 1024**2        # MB
            })
        
        return memory_info
    
    def optimize_memory(self):
        """优化内存使用"""
        # 清理缓存
        if hasattr(self, '_model_cache'):
            self._model_cache.clear()
        
        if hasattr(self, '_get_cached_preprocessing'):
            self._get_cached_preprocessing.cache_clear()
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 强制垃圾回收
        gc.collect()
        
        print("内存优化完成")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用模型列表"""
        return {
            'vehicle_models': list(self.vehicle_models.keys()),
            'water_models': list(self.water_models.keys())
        }
    
    def set_active_models(self, vehicle_model: str, water_model: str) -> bool:
        """设置活跃模型"""
        success = True
        
        if vehicle_model and vehicle_model in self.vehicle_models:
            self.current_vehicle_model = vehicle_model
        else:
            print(f"车辆检测模型不存在: {vehicle_model}")
            success = False
        
        if water_model and water_model in self.water_models:
            self.current_water_model = water_model
        else:
            print(f"水面分割模型不存在: {water_model}")
            success = False
        
        return success
    
    def get_current_models(self) -> Dict[str, Optional[str]]:
        """获取当前活跃的模型"""
        return {
            'vehicle_model': self.current_vehicle_model,
            'water_model': self.current_water_model
        }
    
    def _create_deeplabv3_model(self):
        """创建DeepLabV3模型架构"""
        try:
            import torchvision.models.segmentation as segmentation
            
            # 创建DeepLabV3模型（使用ResNet101作为backbone）
            # 使用1个类别，因为原始模型是单类别的
            model = segmentation.deeplabv3_resnet101(
                weights=None,  # 不使用预训练权重
                num_classes=1   # 1个类别：水面（背景通过阈值处理）
            )
            
            return model
            
        except Exception as e:
            print(f"创建DeepLabV3模型失败: {e}")
            # 如果创建失败，返回简单的水面检测器
            return self._create_simple_water_detector()
    
    def _create_simple_water_detector(self):
        """创建一个简单的水面检测器（当DeepLabV3不可用时）"""
        class SimpleWaterDetector:
            def __init__(self, device):
                self.device = device
            
            def __call__(self, input_tensor):
                # 简单的基于颜色的水面检测
                # 这是一个占位符实现，实际应用中需要更复杂的逻辑
                batch_size, channels, height, width = input_tensor.shape
                
                # 创建一个简单的掩码：检测蓝色区域作为水面
                # 将输入从[-1,1]范围转换回[0,1]
                normalized_input = (input_tensor + 1) / 2
                
                # 提取蓝色通道（假设输入是RGB格式）
                blue_channel = normalized_input[:, 2, :, :]  # 蓝色通道
                green_channel = normalized_input[:, 1, :, :]  # 绿色通道
                red_channel = normalized_input[:, 0, :, :]   # 红色通道
                
                # 简单的蓝色检测：蓝色 > 红色 且 蓝色 > 绿色
                water_mask = (blue_channel > red_channel) & (blue_channel > green_channel) & (blue_channel > 0.3)
                
                # 创建输出格式（模拟DeepLabV3的输出）
                output = torch.zeros(batch_size, 2, height, width, device=input_tensor.device)
                output[:, 0, :, :] = ~water_mask  # 背景
                output[:, 1, :, :] = water_mask   # 水面
                
                return {'out': output}
        
        return SimpleWaterDetector(self.device)
    
