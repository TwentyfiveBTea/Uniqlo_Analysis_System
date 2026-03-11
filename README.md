# Uniqlo Analysis System

基于 Hadoop 的优衣库订单数据分析与可视化系统。

## 项目简介

优衣库分析系统是一个用于分析优衣库订单数据的数据分析平台，采用前后端分离架构：

- **后端**：Python Flask API 服务，提供数据分析和机器学习接口
- **前端**：Vue 3 + ECharts 可视化界面，展示数据分析结果

### 核心功能

| 功能模块 | 说明 |
|---------|------|
| ARIMA 销量预测 | 基于时间序列的销量趋势预测 |
| K-means 用户聚类 | 用户画像分析与分群 |
| Decision Tree 决策树 | 区域销售分析 |
| Apriori 关联规则 | 商品关联性分析 |

## 技术栈

- **后端**：Python 3.8+, Flask, Pandas, NumPy, Scikit-learn, Statsmodels
- **前端**：Vue 3, Vite, ECharts, Element Plus

---

## 环境准备

### 后端依赖 (Python)

#### Windows 安装命令

使用清华镜像源安装：

```cmd
# 创建虚拟环境（可选）
python -m venv venv
venv\Scripts\activate

# 使用镜像源安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

使用阿里镜像源安装：

```cmd
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

#### Mac 安装命令

使用清华镜像源安装：

```bash
# 创建虚拟环境（可选）
python3 -m venv venv
source venv/bin/activate

# 使用镜像源安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

使用阿里镜像源安装：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

#### 后端可选依赖

如需完整的机器学习支持，可以额外安装：

```bash
# 使用镜像源安装
pip install statsmodels>=0.14.0 scikit-learn>=1.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 前端依赖 (Node.js)

#### Windows 安装命令

使用淘宝镜像源：

```cmd
# 进入前端目录
cd uniqlo_frontend

# 使用镜像源安装依赖
npm install --registry=https://registry.npmmirror.com
```

或者使用 nrm 切换镜像：

```cmd
npm config set registry https://registry.npmmirror.com
npm install
```

#### Mac 安装命令

使用淘宝镜像源：

```bash
# 进入前端目录
cd uniqlo_frontend

# 使用镜像源安装依赖
npm install --registry=https://registry.npmmirror.com
```

或者全局配置镜像：

```bash
npm config set registry https://registry.npmmirror.com
cd uniqlo_frontend
npm install
```

---

## 快速启动

### 启动后端服务

```bash
# 后端目录
cd uniqlo_backend

# 启动 Flask 服务（默认端口 5000）
python app.py

# 或使用 gunicorn（生产环境）
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 启动前端服务

```bash
# 前端目录
cd uniqlo_frontend

# 开发模式启动（默认端口 5173）
npm run dev

# 构建生产版本
npm run build

# 预览生产版本
npm run preview
```

### 访问系统

- 前端地址：http://localhost:5173
- 后端 API：http://localhost:5000

---

## 项目结构

```
Uniqlo_Analysis_System/
├── uniqlo_backend/           # 后端服务
│   ├── app.py                # Flask 主应用
│   ├── requirements.txt      # Python 依赖
│   ├── algorithms/          # 算法模块
│   │   ├── arima_trend.py           # ARIMA 预测
│   │   ├── user_portrait_kmeans.py  # K-means 聚类
│   │   ├── distribution_tree.py     # 决策树
│   │   └── market_basket_apriori.py  # Apriori 关联
│   ├── hadoop_tools/         # Hadoop 工具
│   └── data/                 # 数据文件
│
└── uniqlo_frontend/          # 前端应用
    ├── src/
    │   ├── views/            # 页面组件
    │   ├── components/       # 公共组件
    │   └── router/           # 路由配置
    └── package.json          # Node 依赖
```

---

## 常用命令汇总

### 后端命令

| 命令 | 说明 |
|------|------|
| `python app.py` | 启动 Flask 开发服务器 |
| `gunicorn -w 4 -b 0.0.0.0:5000 app:app` | 启动 Gunicorn 生产服务器 |
| `pytest` | 运行测试 |

### 前端命令

| 命令 | 说明 |
|------|------|
| `npm run dev` | 启动开发服务器 |
| `npm run build` | 构建生产版本 |
| `npm run preview` | 预览生产版本 |

---

## 常见问题

### 1. 安装依赖失败

- 检查 Python 版本是否 >= 3.8
- 尝试更换镜像源（清华/阿里/腾讯）
- 确保网络可以访问镜像站点

### 2. 端口被占用

- 后端默认端口 5000，前端默认端口 5173
- 如需更换，可以在启动命令中指定端口

### 3. 前端无法连接后端

- 检查后端服务是否正常运行
- 检查前端配置的 API 地址是否正确

---

## License

MIT License
