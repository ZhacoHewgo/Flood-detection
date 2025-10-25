import React, { useState, useEffect } from 'react';
import { Layout, message, Spin } from 'antd';
import './App.css';

import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import ModelSelector from './components/ModelSelector';
import ResultDisplay from './components/ResultDisplay';
import StatisticsPanel from './components/StatisticsPanel';
import { apiService } from './services/apiService';
import { AnalysisResult, AvailableModels, ModelSelection } from './types';

const { Content, Footer } = Layout;

interface AppState {
  selectedImage: File | null;
  imagePreview: string | null;
  analysisResult: AnalysisResult | null;
  isAnalyzing: boolean;
  availableModels: AvailableModels;
  selectedModels: ModelSelection;
  loading: boolean;
}

const App: React.FC = () => {
  const [state, setState] = useState<AppState>({
    selectedImage: null,
    imagePreview: null,
    analysisResult: null,
    isAnalyzing: false,
    availableModels: { vehicle_models: [], water_models: [] },
    selectedModels: { vehicle_model: '', water_model: '' },
    loading: true
  });

  // 组件挂载时加载可用模型
  useEffect(() => {
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const models = await apiService.getAvailableModels();
      setState(prev => ({
        ...prev,
        availableModels: models,
        selectedModels: {
          vehicle_model: models.vehicle_models[0] || '',
          water_model: models.water_models[0] || ''
        },
        loading: false
      }));
    } catch (error) {
      message.error('加载模型列表失败');
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  const handleImageSelect = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setState(prev => ({
        ...prev,
        selectedImage: file,
        imagePreview: e.target?.result as string,
        analysisResult: null
      }));
    };
    reader.readAsDataURL(file);
  };

  const handleModelChange = (models: ModelSelection) => {
    setState(prev => ({
      ...prev,
      selectedModels: models
    }));
  };

  const handleAnalyze = async () => {
    if (!state.selectedImage) {
      message.warning('请先选择图像文件');
      return;
    }

    if (!state.selectedModels.vehicle_model || !state.selectedModels.water_model) {
      message.warning('请选择检测和分割模型');
      return;
    }

    setState(prev => ({ ...prev, isAnalyzing: true }));

    try {
      const result = await apiService.analyzeImage(
        state.selectedImage,
        state.selectedModels.vehicle_model,
        state.selectedModels.water_model
      );

      setState(prev => ({
        ...prev,
        analysisResult: result,
        isAnalyzing: false
      }));

      message.success('分析完成！');
    } catch (error) {
      message.error('分析失败，请重试');
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
        height: '100vh' 
      }}>
        <Spin size="large" tip="正在加载应用..." />
      </div>
    );
  }

  return (
    <Layout className="app-layout">
      <Header />
      
      <Content className="app-content">
        <div className="content-container">
          {/* 控制面板 */}
          <div className="control-panel">
            <ImageUpload
              onImageSelect={handleImageSelect}
              disabled={state.isAnalyzing}
              selectedImage={state.selectedImage}
            />
            
            <ModelSelector
              availableModels={state.availableModels}
              selectedModels={state.selectedModels}
              onModelChange={handleModelChange}
              disabled={state.isAnalyzing}
            />
            
            <div className="action-buttons">
              <button
                className="analyze-button"
                onClick={handleAnalyze}
                disabled={!state.selectedImage || state.isAnalyzing}
              >
                {state.isAnalyzing ? '分析中...' : '开始分析'}
              </button>
              
              <button
                className="reset-button"
                onClick={handleReset}
                disabled={state.isAnalyzing}
              >
                重置
              </button>
            </div>
          </div>

          {/* 结果显示区域 */}
          <div className="result-area">
            <ResultDisplay
              originalImage={state.imagePreview}
              analysisResult={state.analysisResult}
              isAnalyzing={state.isAnalyzing}
            />
            
            {state.analysisResult && (
              <StatisticsPanel
                statistics={state.analysisResult.statistics}
                vehicles={state.analysisResult.vehicles}
              />
            )}
          </div>
        </div>
      </Content>
      
      <Footer className="app-footer">
        积水车辆检测系统 ©2024 - 基于深度学习的智能分析平台
      </Footer>
    </Layout>
  );
};

export default App;