import axios from 'axios';
import { AnalysisResult, AvailableModels, HealthStatus } from '../types';

// API基础URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 创建axios实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60秒超时
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('API请求:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('API响应:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('API错误:', error.response?.status, error.message);
    
    if (error.response?.status === 503) {
      throw new Error('服务暂时不可用，请稍后重试');
    } else if (error.response?.status === 500) {
      throw new Error('服务器内部错误');
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('请求超时，请检查网络连接');
    }
    
    throw error;
  }
);

export const apiService = {
  // 健康检查
  async checkHealth(): Promise<HealthStatus> {
    const response = await apiClient.get<HealthStatus>('/api/health');
    return response.data;
  },

  // 获取可用模型
  async getAvailableModels(): Promise<AvailableModels> {
    const response = await apiClient.get<AvailableModels>('/api/models');
    return response.data;
  },

  // 分析图像
  async analyzeImage(
    imageFile: File,
    vehicleModel: string,
    waterModel: string
  ): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('vehicle_model', vehicleModel);
    formData.append('water_model', waterModel);

    const response = await apiClient.post<AnalysisResult>('/api/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 120000, // 2分钟超时，因为分析可能需要较长时间
    });

    return response.data;
  },

  // 批量分析图像
  async batchAnalyzeImages(
    imageFiles: File[],
    vehicleModel: string,
    waterModel: string
  ): Promise<any> {
    const formData = new FormData();
    
    imageFiles.forEach(file => {
      formData.append('files', file);
    });
    formData.append('vehicle_model', vehicleModel);
    formData.append('water_model', waterModel);

    const response = await apiClient.post('/api/batch-analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5分钟超时，批量分析需要更长时间
    });

    return response.data;
  },

  // 获取性能信息
  async getPerformanceInfo(): Promise<any> {
    const response = await apiClient.get('/api/performance');
    return response.data;
  },

  // 优化性能
  async optimizePerformance(): Promise<any> {
    const response = await apiClient.post('/api/optimize');
    return response.data;
  },

  // 测试连接
  async testConnection(): Promise<boolean> {
    try {
      const response = await apiClient.get('/');
      return response.status === 200;
    } catch {
      return false;
    }
  },
};