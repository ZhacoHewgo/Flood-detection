"""
FastAPI后端服务 - 完整版
FastAPI Backend Service for Flood Detection Web Application - Complete Version
"""

import os
import io
import base64
import time
import asyncio
import hashlib
import json
import psutil
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import gc
import logging
from datetime import datetime, timedelta

# 导入核心模块
from ...core import (
    ModelManager, ImageProcessor, FloodAnalyzer, VisualizationEngine,
    config_manager
)
from ...core.exceptions import FloodDetectionError, ModelLoadError, InferenceError
from ...core.data_models import Statistics, AnalysisResult


# Pydantic模型定义
class VehicleResult(BaseModel):
    id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    flood_level: str
    overlap_ratio: float


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    vehicles: List[VehicleResult]
    statistics: Dict[str, Any]
    processing_time: float
    result_image_base64: str
    water_coverage_percentage: float
    cache_hit: Optional[bool] = False
    analysis_id: Optional[str] = None


class BatchAnalysisRequest(BaseModel):
    vehicle_model: str
    water_model: str
    task_mode: str = "combined"  # "vehicle_only", "water_only", "combined"


class BatchAnalysisResponse(BaseModel):
    success: bool
    message: str
    total_files: int
    processed_files: int
    failed_files: int
    results: List[Dict[str, Any]]
    total_processing_time: float
    batch_id: str


class ModelsResponse(BaseModel):
    vehicle_models: List[str]
    water_models: List[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    models_loaded: bool
    version: str


class PerformanceResponse(BaseModel):
    cpu_usage: float
    memory_usage: float
    memory_available: float
    cache_size: int
    active_models: Dict[str, bool]
    uptime: float


class ErrorResponse(BaseModel):
    success: bool
    error: str
    error_code: str


class CacheInfo(BaseModel):
    total_entries: int
    memory_usage_mb: float
    hit_rate: float
    oldest_entry: Optional[str] = None


# 创建FastAPI应用 - 完整版
app = FastAPI(
    title="积水车辆检测API - 完整版",
    description="基于深度学习的积水车辆检测分析服务 - 功能完整版",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
model_manager = None
image_processor = None
flood_analyzer = None
viz_engine = None
models_loaded = False
app_start_time = time.time()

# 性能优化设置
executor = ThreadPoolExecutor(max_workers=6)  # 增加线程池大小
analysis_lock = threading.Lock()  # 防止并发分析冲突
cache_lock = threading.Lock()

# 智能缓存系统
class AnalysisCache:
    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, image_data: bytes, vehicle_model: str, water_model: str, task_mode: str) -> str:
        """生成缓存键"""
        content = image_data + vehicle_model.encode() + water_model.encode() + task_mode.encode()
        return hashlib.md5(content).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """获取缓存"""
        with cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                # 检查TTL
                if datetime.now() - entry['timestamp'] < self.ttl:
                    self.access_times[key] = datetime.now()
                    self.hits += 1
                    return entry['data']
                else:
                    # 过期删除
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
            
            self.misses += 1
            return None
    
    def set(self, key: str, data: Dict):
        """设置缓存"""
        with cache_lock:
            # 如果缓存满了，删除最旧的条目
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            self.access_times[key] = datetime.now()
    
    def clear(self):
        """清空缓存"""
        with cache_lock:
            self.cache.clear()
            self.access_times.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        with cache_lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            # 计算内存使用（粗略估计）
            memory_usage = 0
            for entry in self.cache.values():
                memory_usage += len(str(entry).encode('utf-8'))
            
            oldest_entry = None
            if self.access_times:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                oldest_entry = self.access_times[oldest_key].isoformat()
            
            return {
                'total_entries': len(self.cache),
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'hit_rate': hit_rate,
                'oldest_entry': oldest_entry
            }

# 创建缓存实例
analysis_cache = AnalysisCache()


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global model_manager, image_processor, flood_analyzer, viz_engine, models_loaded
    
    try:
        logger.info("正在初始化模型管理器...")
        model_manager = ModelManager()
        
        logger.info("正在初始化图像处理器...")
        image_processor = ImageProcessor()
        
        logger.info("正在初始化洪水分析器...")
        flood_analyzer = FloodAnalyzer()
        
        logger.info("正在初始化可视化引擎...")
        viz_engine = VisualizationEngine()
        
        models_loaded = True
        logger.info("所有模块初始化完成")
        
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        models_loaded = False


@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "积水车辆检测API - 完整版",
        "version": "2.0.0",
        "status": "running" if models_loaded else "initializing"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if models_loaded else "initializing",
        timestamp=time.time(),
        models_loaded=models_loaded,
        version="2.0.0"
    )


@app.get("/models", response_model=ModelsResponse)
async def get_available_models():
    """获取可用模型列表"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        vehicle_models = model_manager.get_available_vehicle_models()
        water_models = model_manager.get_available_water_models()
        
        return ModelsResponse(
            vehicle_models=vehicle_models,
            water_models=water_models
        )
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@app.get("/performance", response_model=PerformanceResponse)
async def get_performance_metrics():
    """获取性能指标"""
    try:
        # 获取系统性能指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # 获取模型状态
        active_models = {}
        if models_loaded and model_manager:
            try:
                active_models = {
                    "vehicle_model_loaded": model_manager.vehicle_model is not None,
                    "water_model_loaded": model_manager.water_model is not None
                }
            except:
                active_models = {"vehicle_model_loaded": False, "water_model_loaded": False}
        
        # 计算运行时间
        uptime = time.time() - app_start_time
        
        return PerformanceResponse(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            memory_available=memory.available / (1024**3),  # GB
            cache_size=len(analysis_cache.cache),
            active_models=active_models,
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"获取性能指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@app.post("/optimize")
async def optimize_performance():
    """性能优化"""
    try:
        # 清理缓存
        analysis_cache.clear()
        
        # 强制垃圾回收
        gc.collect()
        
        # 如果有CUDA，清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        return {"success": True, "message": "性能优化完成"}
    except Exception as e:
        logger.error(f"性能优化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/cache/info", response_model=CacheInfo)
async def get_cache_info():
    """获取缓存信息"""
    try:
        stats = analysis_cache.get_stats()
        return CacheInfo(**stats)
    except Exception as e:
        logger.error(f"获取缓存信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache info: {str(e)}")


@app.delete("/cache")
async def clear_cache():
    """清空缓存"""
    try:
        analysis_cache.clear()
        return {"success": True, "message": "缓存已清空"}
    except Exception as e:
        logger.error(f"清空缓存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


def process_single_image(image_data: bytes, vehicle_model: str, water_model: str, task_mode: str) -> Dict[str, Any]:
    """处理单张图像"""
    try:
        # 检查缓存
        cache_key = analysis_cache._generate_key(image_data, vehicle_model, water_model, task_mode)
        cached_result = analysis_cache.get(cache_key)
        if cached_result:
            cached_result['cache_hit'] = True
            return cached_result
        
        # 解码图像
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # 确保图像是RGB格式
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB
            pass  # 已经是RGB
        else:
            raise ValueError("Unsupported image format")
        
        start_time = time.time()
        
        with analysis_lock:
            # 根据任务模式设置模型
            if task_mode == "vehicle_only":
                model_manager.set_active_models(vehicle_model, None)
            elif task_mode == "water_only":
                model_manager.set_active_models(None, water_model)
            else:  # combined
                model_manager.set_active_models(vehicle_model, water_model)
            
            # 执行分析
            vehicles = []
            water_mask = None
            
            if task_mode in ["vehicle_only", "combined"]:
                vehicles = model_manager.predict_vehicles(image_np)
            
            if task_mode in ["water_only", "combined"]:
                water_mask = model_manager.predict_water(image_np)
            
            # 分析结果
            if task_mode == "combined" and vehicles and water_mask is not None:
                analysis_result = flood_analyzer.analyze_scene(vehicles, water_mask)
            else:
                # 创建简化的分析结果
                from ...core.data_models import Statistics
                
                if water_mask is not None:
                    water_coverage = (np.sum(water_mask > 0) / water_mask.size) * 100
                else:
                    water_coverage = 0.0
                
                stats = Statistics(
                    total_vehicles=len(vehicles) if vehicles else 0,
                    light_flood_count=0,
                    moderate_flood_count=0,
                    severe_flood_count=0,
                    water_coverage_percentage=water_coverage,
                    processing_time=time.time() - start_time
                )
                
                analysis_result = AnalysisResult(
                    vehicles=vehicles if vehicles else [],
                    statistics=stats,
                    water_mask=water_mask
                )
            
            # 生成可视化结果
            result_image = viz_engine.create_visualization(
                image_np, analysis_result.vehicles, analysis_result.water_mask
            )
            
            # 转换为base64
            result_pil = Image.fromarray(result_image)
            buffer = io.BytesIO()
            result_pil.save(buffer, format='PNG')
            result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        processing_time = time.time() - start_time
        
        # 构建响应
        response_data = {
            "success": True,
            "message": "分析完成",
            "vehicles": [
                {
                    "id": i,
                    "bbox": vehicle.bbox.tolist() if hasattr(vehicle.bbox, 'tolist') else list(vehicle.bbox),
                    "confidence": float(vehicle.confidence),
                    "flood_level": vehicle.flood_level if hasattr(vehicle, 'flood_level') else "unknown",
                    "overlap_ratio": float(vehicle.overlap_ratio) if hasattr(vehicle, 'overlap_ratio') else 0.0
                }
                for i, vehicle in enumerate(analysis_result.vehicles)
            ],
            "statistics": {
                "total_vehicles": analysis_result.statistics.total_vehicles,
                "light_flood_count": analysis_result.statistics.light_flood_count,
                "moderate_flood_count": analysis_result.statistics.moderate_flood_count,
                "severe_flood_count": analysis_result.statistics.severe_flood_count,
                "water_coverage_percentage": analysis_result.statistics.water_coverage_percentage,
                "processing_time": processing_time
            },
            "processing_time": processing_time,
            "result_image_base64": result_base64,
            "water_coverage_percentage": analysis_result.statistics.water_coverage_percentage,
            "cache_hit": False,
            "analysis_id": cache_key
        }
        
        # 缓存结果
        analysis_cache.set(cache_key, response_data)
        
        return response_data
        
    except Exception as e:
        logger.error(f"图像处理失败: {str(e)}")
        raise Exception(f"Image processing failed: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    vehicle_model: str = Form(...),
    water_model: str = Form(...),
    task_mode: str = Form(default="combined")
):
    """分析单张图像"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # 验证文件类型
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        # 读取图像数据
        image_data = await file.read()
        
        # 验证文件大小 (最大50MB)
        if len(image_data) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")
        
        # 在线程池中处理图像
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            process_single_image, 
            image_data, 
            vehicle_model, 
            water_model, 
            task_mode
        )
        
        return AnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    files: List[UploadFile] = File(...),
    vehicle_model: str = Form(...),
    water_model: str = Form(...),
    task_mode: str = Form(default="combined")
):
    """批量分析图像"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # 验证文件数量
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Too many files. Maximum is 50 files per batch.")
    
    batch_id = hashlib.md5(f"{time.time()}_{len(files)}".encode()).hexdigest()
    start_time = time.time()
    
    results = []
    processed_files = 0
    failed_files = 0
    
    try:
        # 读取所有文件数据
        file_data_list = []
        for i, file in enumerate(files):
            if not file.content_type or not file.content_type.startswith('image/'):
                failed_files += 1
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid file type",
                    "index": i
                })
                continue
            
            try:
                data = await file.read()
                if len(data) > 50 * 1024 * 1024:
                    failed_files += 1
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "File too large",
                        "index": i
                    })
                    continue
                
                file_data_list.append((i, file.filename, data))
            except Exception as e:
                failed_files += 1
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e),
                    "index": i
                })
        
        # 并行处理图像
        loop = asyncio.get_event_loop()
        
        # 创建任务列表
        tasks = []
        for index, filename, data in file_data_list:
            task = loop.run_in_executor(
                executor,
                process_single_image,
                data,
                vehicle_model,
                water_model,
                task_mode
            )
            tasks.append((index, filename, task))
        
        # 等待所有任务完成
        for index, filename, task in tasks:
            try:
                result = await task
                result['filename'] = filename
                result['index'] = index
                results.append(result)
                processed_files += 1
            except Exception as e:
                failed_files += 1
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e),
                    "index": index
                })
        
        total_processing_time = time.time() - start_time
        
        return BatchAnalysisResponse(
            success=True,
            message=f"批量分析完成: {processed_files}成功, {failed_files}失败",
            total_files=len(files),
            processed_files=processed_files,
            failed_files=failed_files,
            results=results,
            total_processing_time=total_processing_time,
            batch_id=batch_id
        )
        
    except Exception as e:
        logger.error(f"批量分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
async def startup_event():
    """应用启动事件"""
    global model_manager, image_processor, flood_analyzer, viz_engine, models_loaded
    
    try:
        print("正在初始化后端服务...")
        
        # 初始化核心组件
        model_manager = ModelManager()
        image_processor = ImageProcessor()
        flood_analyzer = FloodAnalyzer()
        viz_engine = VisualizationEngine()
        
        # 加载模型
        print("正在加载深度学习模型...")
        success = model_manager.load_models()
        
        if success:
            models_loaded = True
            print("✅ 模型加载成功")
        else:
            print("⚠️ 部分模型加载失败")
            
        print("🚀 后端服务启动完成")
        
    except Exception as e:
        print(f"❌ 后端服务启动失败: {e}")
        models_loaded = False


@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "积水车辆检测API服务",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        timestamp=time.time(),
        models_loaded=models_loaded,
        version="1.0.0"
    )


@app.get("/api/models", response_model=ModelsResponse)
async def get_available_models():
    """获取可用模型列表"""
    try:
        if not models_loaded or model_manager is None:
            raise HTTPException(
                status_code=503,
                detail="模型未加载，服务不可用"
            )
        
        available_models = model_manager.get_available_models()
        
        return ModelsResponse(
            vehicle_models=available_models.get('vehicle_models', []),
            water_models=available_models.get('water_models', [])
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取模型列表失败: {str(e)}"
        )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    vehicle_model: str = Form("YOLOv11 Car Detection"),
    water_model: str = Form("DeepLabV3 Water Segmentation")
):
    """单张图像分析接口"""
    """图像分析接口 - 性能优化版本"""
    try:
        # 检查服务状态
        if not models_loaded or model_manager is None:
            raise HTTPException(
                status_code=503,
                detail="模型未加载，服务不可用"
            )
        
        # 验证文件类型和大小
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="文件类型错误，请上传图像文件"
            )
        
        # 检查文件大小（限制为10MB）
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="文件过大，请上传小于10MB的图像"
            )
        
        start_time = time.time()
        
        # 异步读取图像文件
        image_data = await file.read()
        
        # 生成缓存键
        cache_key = f"{vehicle_model}_{water_model}_{hash(image_data)}"
        
        # 检查缓存
        with cache_lock:
            if cache_key in request_cache:
                cached_result = request_cache[cache_key]
                print(f"使用缓存结果，缓存键: {cache_key[:20]}...")
                return cached_result
        
        # 使用线程池执行CPU密集型任务
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            perform_analysis_optimized, 
            image_data, 
            vehicle_model, 
            water_model
        )
        
        # 添加后台任务进行内存清理
        background_tasks.add_task(cleanup_memory)
        
        # 缓存结果（限制缓存大小）
        with cache_lock:
            if len(request_cache) > 50:  # 限制缓存条目数
                # 删除最旧的缓存项
                oldest_key = next(iter(request_cache))
                del request_cache[oldest_key]
            request_cache[cache_key] = result
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"分析失败: {str(e)}"
        )


def perform_analysis_optimized(
    image_data: bytes, 
    vehicle_model: str, 
    water_model: str
) -> AnalysisResponse:
    """
    优化的图像分析函数
    
    Args:
        image_data: 图像字节数据
        vehicle_model: 车辆检测模型名称
        water_model: 水面分割模型名称
        
    Returns:
        AnalysisResponse: 分析响应
    """
    start_time = time.time()
    
    try:
        # 使用分析锁防止并发冲突
        with analysis_lock:
            # 快速加载图像
            image = load_image_from_bytes_fast(image_data)
            
            # 设置模型
            success = model_manager.set_active_models(vehicle_model, water_model)
            if not success:
                raise ValueError(f"无法设置模型: {vehicle_model}, {water_model}")
            
            # 执行分析
            analysis_result = perform_analysis_fast(image)
            
            # 生成结果图像
            result_image = viz_engine.create_result_image_fast(image, analysis_result)
            result_image_base64 = image_to_base64_fast(result_image)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 构建车辆结果
            vehicles = []
            for vehicle in analysis_result.vehicles:
                vehicles.append(VehicleResult(
                    id=vehicle.vehicle_id,
                    bbox=[
                        vehicle.detection.bbox.x1,
                        vehicle.detection.bbox.y1,
                        vehicle.detection.bbox.x2,
                        vehicle.detection.bbox.y2
                    ],
                    confidence=vehicle.detection.bbox.confidence,
                    flood_level=vehicle.flood_level.value,
                    overlap_ratio=vehicle.overlap_ratio
                ))
            
            # 构建统计信息
            stats = analysis_result.statistics
            statistics = {
                "total_vehicles": stats.total_vehicles,
                "light_flood_count": stats.light_flood_count,
                "moderate_flood_count": stats.moderate_flood_count,
                "severe_flood_count": stats.severe_flood_count,
                "water_coverage_percentage": stats.water_coverage_percentage,
                "processing_time": processing_time
            }
            
            return AnalysisResponse(
                success=True,
                message="分析完成",
                vehicles=vehicles,
                statistics=statistics,
                processing_time=processing_time,
                result_image_base64=result_image_base64,
                water_coverage_percentage=stats.water_coverage_percentage
            )
    
    except Exception as e:
        raise ValueError(f"分析失败: {str(e)}")

def load_image_from_bytes_fast(image_data: bytes) -> np.ndarray:
    """快速从字节数据加载图像"""
    try:
        # 使用更快的图像加载方法
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            # 回退到PIL方法
            pil_image = Image.open(io.BytesIO(image_data))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
        
    except Exception as e:
        raise ValueError(f"图像加载失败: {str(e)}")

def load_image_from_bytes(image_data: bytes) -> np.ndarray:
    """从字节数据加载图像 - 兼容性方法"""
    return load_image_from_bytes_fast(image_data)


def perform_analysis_fast(image: np.ndarray) -> AnalysisResult:
    """执行快速图像分析"""
    try:
        # 车辆检测
        vehicles = model_manager.predict_vehicles(image)
        
        # 水面分割
        water_mask = model_manager.predict_water(image)
        
        # 淹没分析 - 使用批量优化版本
        analysis_result = flood_analyzer.analyze_scene_batch(vehicles, water_mask)
        
        return analysis_result
        
    except Exception as e:
        raise InferenceError("图像分析", str(e))

def perform_analysis(image: np.ndarray) -> AnalysisResult:
    """执行图像分析 - 兼容性方法"""
    return perform_analysis_fast(image)


def image_to_base64_fast(image: np.ndarray) -> str:
    """快速将图像转换为base64字符串"""
    try:
        # 使用OpenCV直接编码，避免PIL转换
        success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        if not success:
            raise ValueError("图像编码失败")
        
        # 编码为base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        raise ValueError(f"图像编码失败: {str(e)}")

def image_to_base64(image: np.ndarray) -> str:
    """将图像转换为base64字符串 - 兼容性方法"""
    return image_to_base64_fast(image)


# 错误处理
@app.exception_handler(FloodDetectionError)
async def flood_detection_error_handler(request, exc: FloodDetectionError):
    """处理自定义异常"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error=exc.message,
            error_code=exc.error_code or "UNKNOWN_ERROR"
        ).dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """处理HTTP异常"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )


def cleanup_memory():
    """后台内存清理任务"""
    try:
        # 清理模型管理器缓存
        if model_manager:
            model_manager.optimize_memory()
        
        # 强制垃圾回收
        gc.collect()
        
        print("后台内存清理完成")
    except Exception as e:
        print(f"内存清理失败: {e}")

@app.get("/api/performance")
async def get_performance_info():
    """获取性能信息"""
    try:
        performance_info = {
            "models_loaded": models_loaded,
            "cache_size": len(request_cache),
            "executor_threads": executor._max_workers
        }
        
        if model_manager:
            performance_info.update(model_manager.get_memory_usage())
        
        return performance_info
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取性能信息失败: {str(e)}"
        )

@app.post("/api/batch-analyze")
async def batch_analyze_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    vehicle_model: str = Form("YOLOv11 Car Detection"),
    water_model: str = Form("DeepLabV3 Water Segmentation")
):
    """批量图像分析接口"""
    try:
        # 检查服务状态
        if not models_loaded or model_manager is None:
            raise HTTPException(
                status_code=503,
                detail="模型未加载，服务不可用"
            )
        
        # 限制批量大小
        if len(files) > 20:
            raise HTTPException(
                status_code=400,
                detail="批量分析最多支持20个文件"
            )
        
        results = []
        start_time = time.time()
        
        for i, file in enumerate(files):
            try:
                # 验证文件
                if not file.content_type.startswith('image/'):
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "文件类型错误",
                        "result": None
                    })
                    continue
                
                if file.size and file.size > 10 * 1024 * 1024:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "文件过大",
                        "result": None
                    })
                    continue
                
                # 分析图像
                image_data = await file.read()
                
                # 使用线程池执行分析
                loop = asyncio.get_event_loop()
                analysis_result = await loop.run_in_executor(
                    executor,
                    perform_analysis_optimized,
                    image_data,
                    vehicle_model,
                    water_model
                )
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "error": None,
                    "result": analysis_result
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e),
                    "result": None
                })
        
        total_time = time.time() - start_time
        successful_count = sum(1 for r in results if r["success"])
        
        # 添加后台清理任务
        background_tasks.add_task(cleanup_memory)
        
        return {
            "success": True,
            "message": f"批量分析完成: {successful_count}/{len(files)} 成功",
            "results": results,
            "total_processing_time": total_time,
            "successful_count": successful_count,
            "total_count": len(files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"批量分析失败: {str(e)}"
        )


@app.post("/api/optimize")
async def optimize_performance():
    """手动触发性能优化"""
    try:
        # 清理缓存
        with cache_lock:
            request_cache.clear()
        
        # 优化模型内存
        if model_manager:
            model_manager.optimize_memory()
        
        # 强制垃圾回收
        gc.collect()
        
        return {
            "success": True,
            "message": "性能优化完成",
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"性能优化失败: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # 开发环境运行 - 性能优化配置
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1,  # 单进程以避免模型重复加载
        loop="asyncio",
        access_log=False  # 禁用访问日志以提高性能
    )