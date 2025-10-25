# 积水车辆检测系统部署指南

## 概述

本文档提供了积水车辆检测系统的完整部署指南，包括桌面版和Web版的部署方法。

## 系统要求

### 硬件要求
- **CPU**: 4核心以上 (推荐8核心)
- **内存**: 8GB以上 (推荐16GB)
- **存储**: 20GB可用空间
- **GPU**: 可选，支持CUDA的NVIDIA GPU可显著提升推理速度

### 软件要求
- **操作系统**: Ubuntu 20.04+, CentOS 8+, macOS 10.15+, Windows 10+
- **Python**: 3.9+
- **Docker**: 20.10+ (Web版部署)
- **Docker Compose**: 2.0+ (Web版部署)

## 桌面版部署

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd flood-vehicle-detection-gui

# 创建Python虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型文件准备

将以下模型文件放置在 `models/` 目录中：
- `yolov11_car_detection.pt` - YOLOv11车辆检测模型
- `rtdetr_car_detection.pt` - RT-DETR车辆检测模型
- `deeplabv3_water.pt` - DeepLabV3水面分割模型
- `yolov11_seg_water.pt` - YOLOv11水面分割模型

### 3. 启动桌面应用

```bash
python main.py desktop
```

## Web版部署

### 方法1: Docker Compose部署 (推荐)

#### 1. 准备环境

```bash
# 确保Docker和Docker Compose已安装
docker --version
docker-compose --version

# 克隆项目
git clone <repository-url>
cd flood-vehicle-detection-gui
```

#### 2. 配置环境变量

```bash
# 复制环境配置文件
cp flood_detection_app/web/backend/.env.example flood_detection_app/web/backend/.env
cp flood_detection_app/web/frontend/.env.example flood_detection_app/web/frontend/.env

# 编辑配置文件
nano flood_detection_app/web/backend/.env
nano flood_detection_app/web/frontend/.env
```

#### 3. 准备模型文件

确保 `models/` 目录包含所需的模型文件。

#### 4. 启动服务

```bash
# 开发环境
docker-compose up -d

# 生产环境 (包含HTTPS)
docker-compose --profile production up -d
```

#### 5. 验证部署

```bash
# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 访问应用
# 开发环境: http://localhost
# 生产环境: https://yourdomain.com
```

### 方法2: 手动部署

#### 1. 后端部署

```bash
# 进入后端目录
cd flood_detection_app/web/backend

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
nano .env

# 启动后端服务
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 2. 前端部署

```bash
# 进入前端目录
cd flood_detection_app/web/frontend

# 安装依赖
npm install

# 配置环境变量
cp .env.example .env
nano .env

# 构建生产版本
npm run build

# 使用nginx或其他web服务器托管build目录
```

## 生产环境配置

### 1. SSL证书配置

#### 自签名证书 (开发环境)
```bash
./scripts/generate-ssl-cert.sh
```

#### 正式证书 (生产环境)
```bash
# 将证书文件放置到nginx/ssl/目录
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem
```

### 2. 域名配置

编辑 `nginx/nginx.conf`，将 `yourdomain.com` 替换为实际域名。

### 3. 防火墙配置

```bash
# Ubuntu/Debian
sudo ufw allow 80
sudo ufw allow 443

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --permanent --add-port=443/tcp
sudo firewall-cmd --reload
```

## 监控和维护

### 1. 健康检查

```bash
# 检查后端健康状态
curl http://localhost:8000/api/health

# 检查前端状态
curl http://localhost/health
```

### 2. 日志管理

```bash
# 查看Docker容器日志
docker-compose logs backend
docker-compose logs frontend

# 查看nginx日志
docker-compose exec nginx tail -f /var/log/nginx/access.log
```

### 3. 性能监控

```bash
# 查看资源使用情况
docker stats

# 查看系统资源
htop
```

### 4. 备份

```bash
# 备份模型文件
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# 备份配置文件
tar -czf config-backup-$(date +%Y%m%d).tar.gz config/ nginx/ *.yml
```

## 故障排除

### 常见问题

#### 1. 模型加载失败
- 检查模型文件是否存在于 `models/` 目录
- 验证模型文件完整性
- 检查文件权限

#### 2. 内存不足
- 增加系统内存
- 调整模型缓存大小
- 使用CPU推理替代GPU推理

#### 3. 网络连接问题
- 检查防火墙设置
- 验证端口是否被占用
- 检查CORS配置

#### 4. Docker部署问题
```bash
# 重新构建镜像
docker-compose build --no-cache

# 清理Docker资源
docker system prune -a

# 查看详细错误信息
docker-compose logs --tail=50
```

### 性能优化

#### 1. 后端优化
- 启用GPU加速 (如果可用)
- 调整worker进程数量
- 配置模型预加载

#### 2. 前端优化
- 启用Gzip压缩
- 配置静态资源缓存
- 使用CDN加速

#### 3. 数据库优化 (如果使用)
- 配置连接池
- 添加适当索引
- 定期清理日志

## 安全建议

1. **定期更新依赖包**
2. **使用强密码和密钥**
3. **启用HTTPS**
4. **配置防火墙**
5. **定期备份数据**
6. **监控系统日志**
7. **限制API访问频率**

## 支持和联系

如遇到部署问题，请：
1. 查看日志文件获取详细错误信息
2. 检查系统资源使用情况
3. 参考故障排除部分
4. 联系技术支持团队

---

**注意**: 本部署指南假设您具备基本的Linux系统管理和Docker使用经验。如需更详细的说明，请参考相关技术文档。