"""
自定义异常类定义
Custom Exception Classes Definition
"""


class FloodDetectionError(Exception):
    """积水检测系统基础异常类"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ModelLoadError(FloodDetectionError):
    """模型加载失败异常"""
    def __init__(self, model_path: str, original_error: str = None):
        message = f"模型加载失败: {model_path}"
        if original_error:
            message += f" - {original_error}"
        super().__init__(message, "MODEL_LOAD_ERROR")
        self.model_path = model_path
        self.original_error = original_error


class ImageProcessingError(FloodDetectionError):
    """图像处理失败异常"""
    def __init__(self, operation: str, file_path: str = None, original_error: str = None):
        message = f"图像处理失败: {operation}"
        if file_path:
            message += f" - 文件: {file_path}"
        if original_error:
            message += f" - {original_error}"
        super().__init__(message, "IMAGE_PROCESSING_ERROR")
        self.operation = operation
        self.file_path = file_path
        self.original_error = original_error


class InferenceError(FloodDetectionError):
    """模型推理失败异常"""
    def __init__(self, model_name: str, original_error: str = None):
        message = f"模型推理失败: {model_name}"
        if original_error:
            message += f" - {original_error}"
        super().__init__(message, "INFERENCE_ERROR")
        self.model_name = model_name
        self.original_error = original_error


class FileOperationError(FloodDetectionError):
    """文件操作失败异常"""
    def __init__(self, operation: str, file_path: str, original_error: str = None):
        message = f"文件操作失败: {operation} - {file_path}"
        if original_error:
            message += f" - {original_error}"
        super().__init__(message, "FILE_OPERATION_ERROR")
        self.operation = operation
        self.file_path = file_path
        self.original_error = original_error


class ConfigurationError(FloodDetectionError):
    """配置错误异常"""
    def __init__(self, config_item: str, original_error: str = None):
        message = f"配置错误: {config_item}"
        if original_error:
            message += f" - {original_error}"
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_item = config_item
        self.original_error = original_error


class ValidationError(FloodDetectionError):
    """数据验证错误异常"""
    def __init__(self, field: str, value: str, reason: str = None):
        message = f"数据验证失败: {field} = {value}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value
        self.reason = reason