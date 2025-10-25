import React from 'react';
import { Card, Statistic, Progress, Table } from 'antd';
import { CarOutlined, ClockCircleOutlined, PercentageOutlined } from '@ant-design/icons';
import { Statistics, VehicleResult } from '../types';

interface StatisticsPanelProps {
  statistics: Statistics;
  vehicles: VehicleResult[];
}

const StatisticsPanel: React.FC<StatisticsPanelProps> = ({
  statistics,
  vehicles
}) => {
  // 淹没等级颜色映射
  const floodLevelColors = {
    light: '#52c41a',    // 绿色
    moderate: '#faad14', // 橙色
    severe: '#f5222d'    // 红色
  };

  // 淹没等级中文映射
  const floodLevelText = {
    light: '轻度',
    moderate: '中度',
    severe: '重度'
  };

  // 车辆详情表格列定义
  const columns = [
    {
      title: '车辆ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: '淹没等级',
      dataIndex: 'flood_level',
      key: 'flood_level',
      render: (level: string) => (
        <span style={{ color: floodLevelColors[level as keyof typeof floodLevelColors] }}>
          {floodLevelText[level as keyof typeof floodLevelText]}
        </span>
      ),
    },
    {
      title: '重叠比例',
      dataIndex: 'overlap_ratio',
      key: 'overlap_ratio',
      render: (ratio: number) => `${(ratio * 100).toFixed(1)}%`,
    },
    {
      title: '检测置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => `${(confidence * 100).toFixed(1)}%`,
    },
  ];

  return (
    <div className="statistics-panel">
      <h3>分析统计</h3>
      
      {/* 主要统计指标 */}
      <div className="main-statistics">
        <Card>
          <Statistic
            title="车辆总数"
            value={statistics.total_vehicles}
            prefix={<CarOutlined />}
            suffix="辆"
          />
        </Card>
        
        <Card>
          <Statistic
            title="积水覆盖率"
            value={statistics.water_coverage_percentage}
            precision={1}
            prefix={<PercentageOutlined />}
            suffix="%"
          />
        </Card>
        
        <Card>
          <Statistic
            title="处理时间"
            value={statistics.processing_time}
            precision={2}
            prefix={<ClockCircleOutlined />}
            suffix="秒"
          />
        </Card>
      </div>

      {/* 淹没等级分布 */}
      <Card title="淹没等级分布" className="flood-distribution">
        <div className="flood-level-stats">
          <div className="flood-level-item">
            <span className="level-label" style={{ color: floodLevelColors.light }}>
              轻度淹没
            </span>
            <Progress
              percent={(statistics.light_flood_count / statistics.total_vehicles) * 100}
              strokeColor={floodLevelColors.light}
              format={() => `${statistics.light_flood_count}辆`}
            />
          </div>
          
          <div className="flood-level-item">
            <span className="level-label" style={{ color: floodLevelColors.moderate }}>
              中度淹没
            </span>
            <Progress
              percent={(statistics.moderate_flood_count / statistics.total_vehicles) * 100}
              strokeColor={floodLevelColors.moderate}
              format={() => `${statistics.moderate_flood_count}辆`}
            />
          </div>
          
          <div className="flood-level-item">
            <span className="level-label" style={{ color: floodLevelColors.severe }}>
              重度淹没
            </span>
            <Progress
              percent={(statistics.severe_flood_count / statistics.total_vehicles) * 100}
              strokeColor={floodLevelColors.severe}
              format={() => `${statistics.severe_flood_count}辆`}
            />
          </div>
        </div>
      </Card>

      {/* 车辆详情表格 */}
      {vehicles.length > 0 && (
        <Card title="车辆详情" className="vehicle-details">
          <Table
            dataSource={vehicles}
            columns={columns}
            rowKey="id"
            pagination={{ pageSize: 5 }}
            size="small"
          />
        </Card>
      )}
    </div>
  );
};

export default StatisticsPanel;