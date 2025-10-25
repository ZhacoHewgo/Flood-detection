import React, { useCallback } from 'react';
import { Upload, message, Button } from 'antd';
import { InboxOutlined, UploadOutlined } from '@ant-design/icons';
import { useDropzone } from 'react-dropzone';

const { Dragger } = Upload;

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  disabled: boolean;
  selectedImage: File | null;
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageSelect,
  disabled,
  selectedImage
}) => {
  // 支持的图像格式
  const acceptedFormats = {
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png'],
    'image/bmp': ['.bmp'],
    'image/tiff': ['.tiff', '.tif']
  };

  // 文件验证
  const validateFile = (file: File): boolean => {
    // 检查文件类型
    const isValidType = Object.keys(acceptedFormats).includes(file.type);
    if (!isValidType) {
      message.error('请选择有效的图像文件 (JPG, PNG, BMP, TIFF)');
      return false;
    }

    // 检查文件大小 (最大10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      message.error('图像文件大小不能超过10MB');
      return false;
    }

    return true;
  };

  // 处理文件选择
  const handleFileSelect = useCallback((file: File) => {
    if (validateFile(file)) {
      onImageSelect(file);
      message.success(`已选择图像: ${file.name}`);
    }
  }, [onImageSelect]);

  // 拖拽处理
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      handleFileSelect(acceptedFiles[0]);
    }
  }, [handleFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFormats,
    multiple: false,
    disabled
  });

  // Ant Design Upload 配置
  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: Object.keys(acceptedFormats).join(','),
    disabled,
    beforeUpload: (file: File) => {
      handleFileSelect(file);
      return false; // 阻止自动上传
    },
    showUploadList: false,
  };

  return (
    <div className="image-upload-container">
      <h3>选择图像</h3>
      
      {/* 拖拽上传区域 */}
      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''} ${disabled ? 'disabled' : ''}`}>
        <input {...getInputProps()} />
        <Dragger {...uploadProps} className="upload-dragger">
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">
            {isDragActive ? '释放文件到此处' : '点击或拖拽图像文件到此区域'}
          </p>
          <p className="ant-upload-hint">
            支持 JPG, PNG, BMP, TIFF 格式，文件大小不超过10MB
          </p>
        </Dragger>
      </div>

      {/* 文件选择按钮 */}
      <div className="upload-button-container">
        <Upload {...uploadProps}>
          <Button 
            icon={<UploadOutlined />} 
            disabled={disabled}
            size="large"
          >
            选择文件
          </Button>
        </Upload>
      </div>

      {/* 已选择文件信息 */}
      {selectedImage && (
        <div className="selected-file-info">
          <h4>已选择文件:</h4>
          <div className="file-details">
            <p><strong>文件名:</strong> {selectedImage.name}</p>
            <p><strong>文件大小:</strong> {(selectedImage.size / 1024 / 1024).toFixed(2)} MB</p>
            <p><strong>文件类型:</strong> {selectedImage.type}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;