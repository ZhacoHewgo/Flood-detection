import React from 'react';
import { Layout, Typography } from 'antd';
import { CarOutlined } from '@ant-design/icons';

const { Header: AntHeader } = Layout;
const { Title } = Typography;

const Header: React.FC = () => {
  return (
    <AntHeader className="app-header">
      <div className="header-content">
        <div className="logo-section">
          <CarOutlined className="logo-icon" />
          <Title level={3} className="app-title">
            积水车辆检测系统
          </Title>
        </div>
        <div className="subtitle">
          基于深度学习的智能分析平台
        </div>
      </div>
    </AntHeader>
  );
};

export default Header;