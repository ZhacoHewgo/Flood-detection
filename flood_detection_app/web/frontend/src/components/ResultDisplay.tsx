import React from 'react';
import { Card, Spin, Empty } from 'antd';
import { AnalysisResult } from '../types';

interface ResultDisplayProps {
  originalImage: string | null;
  analysisResult: AnalysisResult | null;
  isAnalyzing: boolean;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({
  originalImage,
  analysisResult,
  isAnalyzing
}) => {
  return (
    <div className="result-display">
      <div className="image-comparison">
        {/* 原始图像 */}
        <Card title="原始图像" className="image-card">
          {originalImage ? (
            <img 
              src={originalImage} 
              alt="原始图像" 
              className="display-image"
            />
          ) : (
            <Empty 
              description="请选择图像文件"
              className="empty-placeholder"
            />
          )}
        </Card>

        {/* 分析结果图像 */}
        <Card title="分析结果" className="image-card">
          {isAnalyzing ? (
            <div className="analyzing-placeholder">
              <Spin size="large" />
              <p>正在分析图像...</p>
            </div>
          ) : analysisResult ? (
            <img 
              src={`data:image/jpeg;base64,${analysisResult.result_image_base64}`}
              alt="分析结果" 
              className="display-image"
            />
          ) : (
            <Empty 
              description="分析结果将在此显示"
              className="empty-placeholder"
            />
          )}
        </Card>
      </div>
    </div>
  );
};

export default ResultDisplay;