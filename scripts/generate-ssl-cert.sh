#!/bin/bash
# 生成自签名SSL证书脚本 (仅用于开发环境)

set -e

# 创建SSL目录
mkdir -p nginx/ssl

# 生成私钥
openssl genrsa -out nginx/ssl/key.pem 2048

# 生成证书签名请求
openssl req -new -key nginx/ssl/key.pem -out nginx/ssl/cert.csr -subj "/C=CN/ST=Beijing/L=Beijing/O=FloodDetection/OU=IT/CN=localhost"

# 生成自签名证书
openssl x509 -req -days 365 -in nginx/ssl/cert.csr -signkey nginx/ssl/key.pem -out nginx/ssl/cert.pem

# 清理临时文件
rm nginx/ssl/cert.csr

echo "✅ SSL证书已生成到 nginx/ssl/ 目录"
echo "⚠️  注意: 这是自签名证书，仅适用于开发环境"
echo "📝 生产环境请使用正式的SSL证书"