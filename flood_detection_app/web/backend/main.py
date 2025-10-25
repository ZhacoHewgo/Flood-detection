"""
FastAPI后端服务
FastAPI Backend Service for Flood Detection Web Application
"""

import os
import io
import base64
import time
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import gc

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


class ModelsResponse(BaseModel):
    vehicle_models: List[str]
    water_models: List[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    models_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    success: bool
    error: str
    error_code: str


# 创建FastAPI应用 - 性能优化版本
app = FastAPI(
    title="积水车辆检测API",
    description="基于深度学习的积水车辆检测分析服务 - 性能优化版",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model_manager = None
image_processor = None
flood_analyzer = None
viz_engine = None
models_loaded = False

# 性能优化设置
executor = ThreadPoolExecutor(max_workers=4)  # 用于CPU密集型任务
analysis_lock = threading.Lock()  # 防止并发分析冲突
request_cache = {}  # 简单的请求缓存
cache_lock = threading.Lock()


@app.on_event("startup")
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