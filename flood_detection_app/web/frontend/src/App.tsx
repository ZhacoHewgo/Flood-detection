import React, { useState, useEffect, useCallback } from 'react';
import { Upload, Button, Select, Card, Row, Col, Statistic, Progress, Alert, Tabs, Switch, Space, Spin } from 'antd';
import { InboxOutlined, PlayCircleOutlined, ClearOutlined, DownloadOutlined, MonitorOutlined } from '@ant-design/icons';
import './App.css';
import BatchUpload from './components/BatchUpload';
import PerformanceMonitor from './components/PerformanceMonitor';

const { Dragger } = Upload;
const { Option } = Select;
const { TabPane } = Tabs;

// 完整的类型定义
interface VehicleResult {
  id: number;
  bbox: [number, number, number, number];
  confidence: number;
  flood_level: string;
  overlap_ratio: number;
}

interface Statistics {
  total_vehicles: number;
  light_flood_count: number;
  moderate_flood_count: number;
  severe_flood_count: number;
  water_coverage_percentage: number;
  processing_time: number;
}

interface AnalysisResult {
  success: boolean;
  message: string;
  vehicles: VehicleResult[];
  statistics: Statistics;
  processing_time: number;
  result_image_base64: string;
  water_coverage_percentage: number;
  cache_hit?: boolean;
  analysis_id?: string;
}

interface BatchResult {
  filename: string;
  success: boolean;
  error?: string;
  index: number;
  vehicles?: VehicleResult[];
  statistics?: Statistics;
  result_image_base64?: string;
}

interface BatchAnalysisResult {
  success: boolean;
  message: string;
  total_files: number;
  processed_files: number;
  failed_files: number;
  results: BatchResult[];
  total_processing_time: number;
  batch_id: string;
}

interface AvailableModels {
  vehicle_models: string[];
  water_models: string[];
}

interface ModelSelection {
  vehicle_model: string;
  water_model: string;
}

interface PerformanceMetrics {
  cpu_usage: number;
  memory_usage: number;
  memory_available: number;
  cache_size: number;
  active_models: Record<string, boolean>;
  uptime: number;
}

interface AppState {
  // 单张分析相关
  selectedImage: File | null;
  imagePreview: string | null;
  analysisResult: AnalysisResult | null;
  isAnalyzing: boolean;
  
  // 批量分析相关
  batchFiles: File[];
  batchResults: BatchAnalysisResult | null;
  isBatchAnalyzing: boolean;
  
  // 模型和配置
  availableModels: AvailableModels;
  selectedModels: ModelSelection;
  taskMode: string;
  
  // UI状态
  loading: boolean;
  activeTab: string;
  showPerformanceMonitor: boolean;
  
  // 性能监控
  performanceMetrics: PerformanceMetrics | null;
}

const App: React.FC = () => {
  const [state, setState] = useState<AppState>({
    // 单张分析相关
    selectedImage: null,
    imagePreview: null,
    analysisResult: null,
    isAnalyzing: false,
    
    // 批量分析相关
    batchFiles: [],
    batchResults: null,
    isBatchAnalyzing: false,
    
    // 模型和配置
    availableModels: { vehicle_models: [], water_models: [] },
    selectedModels: { vehicle_model: '', water_model: '' },
    taskMode: 'combined',
    
    // UI状态
    loading: true,
    activeTab: 'single',
    showPerformanceMonitor: false,
    
    // 性能监控
    performanceMetrics: null
  });

  // API基础URL
  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // 组件挂载时初始化
  useEffect(() => {
    loadAvailableModels();
    loadPerformanceMetrics();
  }, []);

  // 加载可用模型
  const loadAvailableModels = async () => {
    try {
      const response = await fetch(`${API_BASE}/models`);
      if (response.ok) {
        const models = await response.json();
        setState(prev => ({
          ...prev,
          availableModels: models,
          selectedModels: {
            vehicle_model: models.vehicle_models[0] || '',
            water_model: models.water_models[0] || ''
          },
          loading: false
        }));
      } else {
        throw new Error('Failed to load models');
      }
    } catch (error) {
      console.error('Error loading models:', error);
      // 使用默认模型作为后备
      setState(prev => ({
        ...prev,
        availableModels: {
          vehicle_models: ['YOLOv11 Car Detection', 'RT-DETR Car Detection'],
          water_models: ['DeepLabV3 Water Segmentation', 'YOLOv11 Water Segmentation']
        },
        selectedModels: {
          vehicle_model: 'YOLOv11 Car Detection',
          water_model: 'DeepLabV3 Water Segmentation'
        },
        loading: false
      }));
    }
  };

  // 加载性能指标
  const loadPerformanceMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/performance`);
      if (response.ok) {
        const metrics = await response.json();
        setState(prev => ({
          ...prev,
          performanceMetrics: metrics
        }));
      }
    } catch (error) {
      console.error('Error loading performance metrics:', error);
    }
  };

  // 单张图像上传处理
  const handleSingleImageUpload = (info: any) => {
    const { file } = info;
    if (file.status === 'done' || file.originFileObj) {
      const fileObj = file.originFileObj || file;
      const reader = new FileReader();
      reader.onload = (e) => {
        setState(prev => ({
          ...prev,
          selectedImage: fileObj,
          imagePreview: e.target?.result as string,
          analysisResult: null
        }));
      };
      reader.readAsDataURL(fileObj);
    }
  };

  // 批量文件上传处理
  const handleBatchUpload = (fileList: File[]) => {
    setState(prev => ({
      ...prev,
      batchFiles: fileList,
      batchResults: null
    }));
  };

  // 模型选择处理
  const handleModelChange = (field: keyof ModelSelection, value: string) => {
    setState(prev => ({
      ...prev,
      selectedModels: {
        ...prev.selectedModels,
        [field]: value
      }
    }));
  };

  // 任务模式切换
  const handleTaskModeChange = (value: string) => {
    setState(prev => ({
      ...prev,
      taskMode: value
    }));
  };

  // 单张图像分析
  const handleSingleAnalysis = async () => {
    if (!state.selectedImage) return;

    setState(prev => ({ ...prev, isAnalyzing: true, analysisResult: null }));

    try {
      const formData = new FormData();
      formData.append('file', state.selectedImage);
      formData.append('vehicle_model', state.selectedModels.vehicle_model);
      formData.append('water_model', state.selectedModels.water_model);
      formData.append('task_mode', state.taskMode);

      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setState(prev => ({ ...prev, analysisResult: result }));
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Analysis failed');
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      alert(`分析失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setState(prev => ({ ...prev, isAnalyzing: false }));
    }
  };

  // 批量分析
  const handleBatchAnalysis = async (files: File[]) => {
    setState(prev => ({ ...prev, isBatchAnalyzing: true, batchResults: null }));

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('vehicle_model', state.selectedModels.vehicle_model);
      formData.append('water_model', state.selectedModels.water_model);
      formData.append('task_mode', state.taskMode);

      const response = await fetch(`${API_BASE}/analyze/batch`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setState(prev => ({ ...prev, batchResults: result }));
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Batch analysis failed');
      }
    } catch (error) {
      console.error('Batch analysis failed:', error);
      alert(`批量分析失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setState(prev => ({ ...prev, isBatchAnalyzing: false }));
    }
  };

  // 清除单张分析结果
  const handleClearSingle = () => {
    setState(prev => ({
      ...prev,
      selectedImage: null,
      imagePreview: null,
      analysisResult: null
    }));
  };

  // 导出单张分析结果
  const handleExportSingle = () => {
    if (!state.analysisResult) return;

    const dataStr = JSON.stringify(state.analysisResult, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `analysis_result_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // 标签页切换
  const handleTabChange = (key: string) => {
    setState(prev => ({ ...prev, activeTab: key }));
  };

  // 性能监控开关
  const togglePerformanceMonitor = () => {
    setState(prev => ({ ...prev, showPerformanceMonitor: !prev.showPerformanceMonitor }));
  };

  if (state.loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="app-header">
        <h1>🌊 积水车辆检测系统 - 完整版</h1>
        <Space>
          <Button 
            icon={<MonitorOutlined />}
            onClick={togglePerformanceMonitor}
            type={state.showPerformanceMonitor ? 'primary' : 'default'}
          >
            性能监控
          </Button>
        </Space>
      </div>

      {state.showPerformanceMonitor && (
        <div style={{ marginBottom: 16 }}>
          <PerformanceMonitor apiBase={API_BASE} autoRefresh={true} />
        </div>
      )}

      {/* 模型配置面板 */}
      <Card title="模型配置" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={8}>
            <div>
              <label>车辆检测模型:</label>
              <Select
                value={state.selectedModels.vehicle_model}
                onChange={(value) => handleModelChange('vehicle_model', value)}
                style={{ width: '100%', marginTop: 8 }}
              >
                {state.availableModels.vehicle_models.map(model => (
                  <Option key={model} value={model}>{model}</Option>
                ))}
              </Select>
            </div>
          </Col>
          <Col span={8}>
            <div>
              <label>水面分割模型:</label>
              <Select
                value={state.selectedModels.water_model}
                onChange={(value) => handleModelChange('water_model', value)}
                style={{ width: '100%', marginTop: 8 }}
              >
                {state.availableModels.water_models.map(model => (
                  <Option key={model} value={model}>{model}</Option>
                ))}
              </Select>
            </div>
          </Col>
          <Col span={8}>
            <div>
              <label>任务模式:</label>
              <Select
                value={state.taskMode}
                onChange={handleTaskModeChange}
                style={{ width: '100%', marginTop: 8 }}
              >
                <Option value="combined">组合分析</Option>
                <Option value="vehicle_only">仅车辆检测</Option>
                <Option value="water_only">仅水面分割</Option>
              </Select>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 主要功能标签页 */}
      <Tabs activeKey={state.activeTab} onChange={handleTabChange}>
        <TabPane tab="单张分析" key="single">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="图像上传">
                <Dragger
                  name="file"
                  accept="image/*"
                  beforeUpload={() => false}
                  onChange={handleSingleImageUpload}
                  showUploadList={false}
                >
                  <p className="ant-upload-drag-icon">
                    <InboxOutlined />
                  </p>
                  <p className="ant-upload-text">点击或拖拽图像文件到此区域上传</p>
                  <p className="ant-upload-hint">支持 JPG, PNG, BMP 等格式</p>
                </Dragger>

                {state.imagePreview && (
                  <div style={{ marginTop: 16, textAlign: 'center' }}>
                    <img 
                      src={state.imagePreview} 
                      alt="预览" 
                      style={{ maxWidth: '100%', maxHeight: 300, objectFit: 'contain' }}
                    />
                  </div>
                )}

                <div style={{ marginTop: 16 }}>
                  <Space>
                    <Button 
                      type="primary" 
                      icon={<PlayCircleOutlined />}
                      onClick={handleSingleAnalysis}
                      disabled={!state.selectedImage || state.isAnalyzing}
                      loading={state.isAnalyzing}
                    >
                      开始分析
                    </Button>
                    <Button 
                      icon={<ClearOutlined />}
                      onClick={handleClearSingle}
                      disabled={state.isAnalyzing}
                    >
                      清除
                    </Button>
                    {state.analysisResult && (
                      <Button 
                        icon={<DownloadOutlined />}
                        onClick={handleExportSingle}
                      >
                        导出结果
                      </Button>
                    )}
                  </Space>
                </div>
              </Card>
            </Col>

            <Col span={12}>
              <Card title="分析结果">
                {state.analysisResult ? (
                  <div>
                    {state.analysisResult.result_image_base64 && (
                      <div style={{ textAlign: 'center', marginBottom: 16 }}>
                        <img 
                          src={`data:image/png;base64,${state.analysisResult.result_image_base64}`}
                          alt="分析结果"
                          style={{ maxWidth: '100%', maxHeight: 300, objectFit: 'contain' }}
                        />
                      </div>
                    )}

                    <Row gutter={16}>
                      <Col span={12}>
                        <Statistic 
                          title="检测车辆" 
                          value={state.analysisResult.statistics.total_vehicles} 
                          suffix="辆"
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic 
                          title="水面覆盖率" 
                          value={state.analysisResult.statistics.water_coverage_percentage} 
                          suffix="%" 
                          precision={1}
                        />
                      </Col>
                    </Row>

                    <Row gutter={16} style={{ marginTop: 16 }}>
                      <Col span={8}>
                        <Statistic 
                          title="轻度积水" 
                          value={state.analysisResult.statistics.light_flood_count} 
                          valueStyle={{ color: '#52c41a' }}
                        />
                      </Col>
                      <Col span={8}>
                        <Statistic 
                          title="中度积水" 
                          value={state.analysisResult.statistics.moderate_flood_count} 
                          valueStyle={{ color: '#faad14' }}
                        />
                      </Col>
                      <Col span={8}>
                        <Statistic 
                          title="重度积水" 
                          value={state.analysisResult.statistics.severe_flood_count} 
                          valueStyle={{ color: '#ff4d4f' }}
                        />
                      </Col>
                    </Row>

                    <div style={{ marginTop: 16 }}>
                      <Statistic 
                        title="处理时间" 
                        value={state.analysisResult.processing_time} 
                        suffix="秒" 
                        precision={2}
                      />
                      {state.analysisResult.cache_hit && (
                        <Alert 
                          message="使用了缓存结果" 
                          type="info" 
                          size="small" 
                          style={{ marginTop: 8 }}
                        />
                      )}
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
                    {state.isAnalyzing ? '分析中...' : '请上传图像并开始分析'}
                  </div>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="批量分析" key="batch">
          <BatchUpload
            files={state.batchFiles}
            onFilesChange={handleBatchUpload}
            onAnalyze={handleBatchAnalysis}
            isAnalyzing={state.isBatchAnalyzing}
            results={state.batchResults}
          />
        </TabPane>
      </Tabs>
    </div>
  );
};

export default App;
      selectedModels: {
        ...prev.selectedModels,
        [field]: value
      }
    }));
  };

  const handleAnalyze = async () => {
    if (!state.selectedImage) {
      alert('请先选择图像文件');
      return;
    }

    setState(prev => ({ ...prev, isAnalyzing: true }));

    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // 模拟分析结果
      const mockResult: AnalysisResult = {
        success: true,
        message: '分析完成',
        vehicles: [
          {
            id: 1,
            bbox: [100, 100, 200, 200],
            confidence: 0.95,
            flood_level: 'moderate',
            overlap_ratio: 0.3
          }
        ],
        statistics: {
          total_vehicles: 1,
          light_flood_count: 0,
          moderate_flood_count: 1,
          severe_flood_count: 0,
          water_coverage_percentage: 25.5,
          processing_time: 2.5
        },
        processing_time: 2.5,
        result_image_base64: '',
        water_coverage_percentage: 25.5
      };

      setState(prev => ({
        ...prev,
        analysisResult: mockResult,
        isAnalyzing: false
      }));

      alert('分析完成！');
    } catch (error) {
      alert('分析失败，请重试');
      setState(prev => ({ ...prev, isAnalyzing: false }));
    }
  };

  const handleReset = () => {
    setState(prev => ({
      ...prev,
      selectedImage: null,
      imagePreview: null,
      analysisResult: null
    }));
  };

  if (state.loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        fontSize: '18px'
      }}>
        正在加载应用...
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌊 积水车辆检测系统</h1>
        <p>基于深度学习的智能分析平台</p>
      </header>
      
      <main className="app-main">
        <div className="control-panel">
          <div className="upload-section">
            <h3>选择图像</h3>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              disabled={state.isAnalyzing}
              className="file-input"
            />
            {state.selectedImage && (
              <div className="file-info">
                <p>已选择: {state.selectedImage.name}</p>
                <p>大小: {(state.selectedImage.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            )}
          </div>

          <div className="model-section">
            <h3>模型选择</h3>
            <div className="model-selector">
              <label>
                车辆检测模型:
                <select
                  value={state.selectedModels.vehicle_model}
                  onChange={(e) => handleModelChange('vehicle_model', e.target.value)}
                  disabled={state.isAnalyzing}
                >
                  {state.availableModels.vehicle_models.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </label>
              
              <label>
                水面分割模型:
                <select
                  value={state.selectedModels.water_model}
                  onChange={(e) => handleModelChange('water_model', e.target.value)}
                  disabled={state.isAnalyzing}
                >
                  {state.availableModels.water_models.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <div className="action-section">
            <button
              onClick={handleAnalyze}
              disabled={!state.selectedImage || state.isAnalyzing}
              className="analyze-btn"
            >
              {state.isAnalyzing ? '分析中...' : '开始分析'}
            </button>
            
            <button
              onClick={handleReset}
              disabled={state.isAnalyzing}
              className="reset-btn"
            >
              重置
            </button>
          </div>
        </div>

        <div className="result-panel">
          <div className="image-display">
            <div className="image-container">
              <h3>原始图像</h3>
              {state.imagePreview ? (
                <img src={state.imagePreview} alt="原始图像" className="display-image" />
              ) : (
                <div className="placeholder">请选择图像文件</div>
              )}
            </div>

            <div className="image-container">
              <h3>分析结果</h3>
              {state.isAnalyzing ? (
                <div className="placeholder">正在分析图像...</div>
              ) : state.analysisResult ? (
                <div className="result-content">
                  <p>分析完成！</p>
                  <div className="stats">
                    <p>检测到车辆: {state.analysisResult.statistics.total_vehicles} 辆</p>
                    <p>水覆盖率: {state.analysisResult.statistics.water_coverage_percentage.toFixed(1)}%</p>
                    <p>处理时间: {state.analysisResult.statistics.processing_time.toFixed(2)}秒</p>
                  </div>
                </div>
              ) : (
                <div className="placeholder">分析结果将在此显示</div>
              )}
            </div>
          </div>

          {state.analysisResult && (
            <div className="statistics-panel">
              <h3>详细统计</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="label">总车辆数:</span>
                  <span className="value">{state.analysisResult.statistics.total_vehicles}</span>
                </div>
                <div className="stat-item">
                  <span className="label">轻度积水:</span>
                  <span className="value">{state.analysisResult.statistics.light_flood_count}</span>
                </div>
                <div className="stat-item">
                  <span className="label">中度积水:</span>
                  <span className="value">{state.analysisResult.statistics.moderate_flood_count}</span>
                </div>
                <div className="stat-item">
                  <span className="label">重度积水:</span>
                  <span className="value">{state.analysisResult.statistics.severe_flood_count}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
      
      <footer className="app-footer">
        积水车辆检测系统 ©2024 - 基于深度学习的智能分析平台
      </footer>
    </div>
  );
};

export default App;