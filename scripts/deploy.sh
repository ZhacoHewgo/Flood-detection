#!/bin/bash
# 自动化部署脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 检查模型文件
check_models() {
    log_info "检查模型文件..."
    
    local models_dir="./models"
    local required_models=(
        "yolov11_car_detection.pt"
        "rtdetr_car_detection.pt"
        "deeplabv3_water.pt"
        "yolov11_seg_water.pt"
    )
    
    if [ ! -d "$models_dir" ]; then
        log_error "模型目录不存在: $models_dir"
        exit 1
    fi
    
    for model in "${required_models[@]}"; do
        if [ ! -f "$models_dir/$model" ]; then
            log_error "缺少模型文件: $model"
            exit 1
        fi
    done
    
    log_success "模型文件检查完成"
}

# 准备环境配置
prepare_env() {
    log_info "准备环境配置..."
    
    # 后端环境配置
    if [ ! -f "flood_detection_app/web/backend/.env" ]; then
        log_info "创建后端环境配置文件..."
        cp flood_detection_app/web/backend/.env.example flood_detection_app/web/backend/.env
    fi
    
    # 前端环境配置
    if [ ! -f "flood_detection_app/web/frontend/.env" ]; then
        log_info "创建前端环境配置文件..."
        cp flood_detection_app/web/frontend/.env.example flood_detection_app/web/frontend/.env
    fi
    
    log_success "环境配置准备完成"
}

# 构建镜像
build_images() {
    log_info "构建Docker镜像..."
    
    docker-compose build --no-cache
    
    log_success "镜像构建完成"
}

# 启动服务
start_services() {
    local profile=${1:-""}
    
    log_info "启动服务..."
    
    if [ "$profile" = "production" ]; then
        log_info "使用生产环境配置..."
        docker-compose --profile production up -d
    else
        log_info "使用开发环境配置..."
        docker-compose up -d
    fi
    
    log_success "服务启动完成"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/api/health &> /dev/null; then
            log_success "后端服务已就绪"
            break
        fi
        
        log_info "等待后端服务启动... ($attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "后端服务启动超时"
        exit 1
    fi
    
    # 检查前端服务
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost/ &> /dev/null; then
            log_success "前端服务已就绪"
            break
        fi
        
        log_info "等待前端服务启动... ($attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "前端服务启动超时"
        exit 1
    fi
}

# 显示服务状态
show_status() {
    log_info "服务状态:"
    docker-compose ps
    
    echo ""
    log_info "访问地址:"
    echo "  前端应用: http://localhost"
    echo "  后端API: http://localhost:8000"
    echo "  API文档: http://localhost:8000/docs"
    echo "  健康检查: http://localhost:8000/api/health"
}

# 清理资源
cleanup() {
    log_info "清理Docker资源..."
    
    docker-compose down
    docker system prune -f
    
    log_success "清理完成"
}

# 显示帮助信息
show_help() {
    echo "积水车辆检测系统部署脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  start [dev|prod]  启动服务 (默认: dev)"
    echo "  stop              停止服务"
    echo "  restart [dev|prod] 重启服务"
    echo "  status            显示服务状态"
    echo "  logs [service]    查看日志"
    echo "  cleanup           清理Docker资源"
    echo "  help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start          # 启动开发环境"
    echo "  $0 start prod     # 启动生产环境"
    echo "  $0 logs backend   # 查看后端日志"
}

# 主函数
main() {
    local command=${1:-"help"}
    local environment=${2:-"dev"}
    
    case $command in
        "start")
            check_dependencies
            check_models
            prepare_env
            build_images
            
            if [ "$environment" = "prod" ] || [ "$environment" = "production" ]; then
                start_services "production"
            else
                start_services
            fi
            
            wait_for_services
            show_status
            ;;
        "stop")
            log_info "停止服务..."
            docker-compose down
            log_success "服务已停止"
            ;;
        "restart")
            log_info "重启服务..."
            docker-compose down
            
            if [ "$environment" = "prod" ] || [ "$environment" = "production" ]; then
                start_services "production"
            else
                start_services
            fi
            
            wait_for_services
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            local service=${2:-""}
            if [ -n "$service" ]; then
                docker-compose logs -f "$service"
            else
                docker-compose logs -f
            fi
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"