import React from 'react';
import { Select, Card } from 'antd';
import { AvailableModels, ModelSelection } from '../types';

const { Option } = Select;

interface ModelSelectorProps {
  availableModels: AvailableModels;
  selectedModels: ModelSelection;
  onModelChange: (models: ModelSelection) => void;
  disabled: boolean;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  availableModels,
  selectedModels,
  onModelChange,
  disabled
}) => {
  const handleVehicleModelChange = (value: string) => {
    onModelChange({
      ...selectedModels,
      vehicle_model: value
    });
  };

  const handleWaterModelChange = (value: string) => {
    onModelChange({
      ...selectedModels,
      water_model: value
    });
  };

  return (
    <Card title="模型选择" className="model-selector">
      <div className="model-select-group">
        <div className="model-select-item">
          <label>车辆检测模型:</label>
          <Select
            value={selectedModels.vehicle_model}
            onChange={handleVehicleModelChange}
            disabled={disabled}
            style={{ width: '100%' }}
            placeholder="选择车辆检测模型"
          >
            {availableModels.vehicle_models.map(model => (
              <Option key={model} value={model}>
                {model}
              </Option>
            ))}
          </Select>
        </div>

        <div className="model-select-item">
          <label>水面分割模型:</label>
          <Select
            value={selectedModels.water_model}
            onChange={handleWaterModelChange}
            disabled={disabled}
            style={{ width: '100%' }}
            placeholder="选择水面分割模型"
          >
            {availableModels.water_models.map(model => (
              <Option key={model} value={model}>
                {model}
              </Option>
            ))}
          </Select>
        </div>
      </div>
    </Card>
  );
};

export default ModelSelector;