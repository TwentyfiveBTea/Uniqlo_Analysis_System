<template>
  <div class="regional-heatmap">
    <!-- 控制栏 -->
    <div class="control-bar">
      <div class="control-group">
        <label class="control-label">决策树深度</label>
        <el-slider v-model="maxDepth" :min="3" :max="10" :marks="{3: '3', 5: '5', 7: '7', 10: '10'}" @change="loadAnalysis" />
      </div>
      <el-button type="primary" size="large" @click="loadAnalysis" :loading="loading">
        开始分析
      </el-button>
    </div>

    <!-- 指标卡片 -->
    <div class="metrics-grid">
      <div class="metric-card" v-for="item in metricItems" :key="item.key">
        <div class="metric-label">{{ item.label }}</div>
        <div class="metric-value">{{ item.value }}</div>
        <div class="metric-desc">{{ item.desc }}</div>
      </div>
    </div>

    <!-- 图表区域 -->
    <div class="charts-grid">
      <div class="chart-section">
        <div class="section-header">
          <h3>特征重要性</h3>
          <span class="tag">Decision Tree</span>
        </div>
        <div ref="featureChartRef" class="chart-container"></div>
      </div>
      <div class="chart-section wide">
        <div class="section-header">
          <h3>区域-品类销量热力图</h3>
        </div>
        <div ref="heatmapChartRef" class="chart-container"></div>
      </div>
    </div>

    <!-- 铺货建议 -->
    <div class="table-section">
      <div class="section-header">
        <h3>铺货建议</h3>
      </div>
      <el-table :data="insights" stripe size="large">
        <el-table-column prop="priority" label="优先级" width="100">
          <template #default="{ row }">
            <span class="priority-tag" :class="row.priority">{{ row.priority }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="region" label="地区" width="100" />
        <el-table-column prop="season" label="季节" width="80" />
        <el-table-column prop="category" label="品类" width="100" />
        <el-table-column prop="price_range" label="价格区间" width="100" />
        <el-table-column label="预期销量" width="100">
          <template #default="{ row }">
            <span class="number">{{ row.expected_sales?.toFixed(0) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="recommendation" label="建议" />
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'

const loading = ref(false)
const maxDepth = ref(5)
const metrics = ref({})
const featureImportance = ref({})
const insights = ref([])
const rules = ref([])
const featureChartRef = ref(null)
const heatmapChartRef = ref(null)
let featureChart = null
let heatmapChart = null

const metricItems = computed(() => [
  { key: 'rmse', label: 'RMSE', value: metrics.value.rmse?.toFixed(2) || '-', desc: '均方根误差' },
  { key: 'mae', label: 'MAE', value: metrics.value.mae?.toFixed(2) || '-', desc: '平均绝对误差' },
  { key: 'r2', label: 'R²', value: metrics.value.r2_score?.toFixed(3) || '-', desc: '决定系数' },
  { key: 'depth', label: '树深度', value: metrics.value.tree_depth || '-', desc: '决策树深度' },
])

const loadAnalysis = async () => {
  loading.value = true
  try {
    const res = await axios.get('/api/decisiontree/analysis', {
      params: { max_depth: maxDepth.value }
    })
    const data = res.data
    
    metrics.value = data.metrics || {}
    featureImportance.value = data.feature_importance || {}
    insights.value = data.insights || []
    rules.value = data.rules || []
    
    updateCharts()
  } catch (e) {
    console.error('Failed to load analysis:', e)
  } finally {
    loading.value = false
  }
}

const updateCharts = () => {
  nextTick(() => {
    if (featureChartRef.value) {
      if (featureChart) featureChart.dispose()
      featureChart = echarts.init(featureChartRef.value)
      
      const sortedFeatures = Object.entries(featureImportance.value)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
      
      featureChart.setOption({
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#fff', borderColor: '#e5e7eb', textStyle: { color: '#1a1a1a' } },
        grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
        xAxis: { type: 'value', axisLabel: { color: '#6b7280' } },
        yAxis: { type: 'category', data: sortedFeatures.map(f => {
          let name = f[0].replace('Region_', '').replace('Season_', '').replace('Category_', '').replace('Price_Range_', '')
          return name
        }) },
        series: [{
          type: 'bar',
          data: sortedFeatures.map(f => f[1]),
          itemStyle: { color: '#1a1a1a' }
        }]
      })
    }
    
    if (heatmapChartRef.value) {
      if (heatmapChart) heatmapChart.dispose()
      heatmapChart = echarts.init(heatmapChartRef.value)
      
      // 获取唯一的地区和品类
      const regions = [...new Set(insights.value.map(i => i.region))]
      const categories = [...new Set(insights.value.map(i => i.category))]
      
      // 过滤数据，只显示销量较高的建议
      const topInsights = insights.value
        .sort((a, b) => b.expected_sales - a.expected_sales)
        .slice(0, 20)  // 只显示前20个
      
      const heatmapData = []
      topInsights.forEach(item => {
        const x = regions.indexOf(item.region)
        const y = categories.indexOf(item.category)
        if (x >= 0 && y >= 0) {
          heatmapData.push([x, y, item.expected_sales])
        }
      })
      
      // 确保有数据
      if (regions.length === 0 || categories.length === 0) {
        return
      }
      
      heatmapChart.setOption({
        tooltip: { 
          position: 'top', 
          backgroundColor: '#fff', 
          borderColor: '#e5e7eb', 
          textStyle: { color: '#1a1a1a' },
          formatter: (params) => {
            if (params.data) {
              const [x, y, value] = params.data
              return `${regions[x]} - ${categories[y]}<br/>预期销量: ${value.toFixed(0)}`
            }
            return ''
          }
        },
        grid: { height: '65%', top: '10%', left: '15%', right: '10%' },
        xAxis: { 
          type: 'category', 
          data: regions, 
          splitArea: { show: true }, 
          axisLabel: { color: '#6b7280', rotate: 30, fontSize: 11 }
        },
        yAxis: { 
          type: 'category', 
          data: categories, 
          splitArea: { show: true }, 
          axisLabel: { color: '#6b7280', fontSize: 11 }
        },
        visualMap: {
          min: 0,
          max: Math.max(...topInsights.map(i => i.expected_sales), 1),
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: '0%',
          inRange: { color: ['#f3f4f6', '#9ca3af', '#4b5563', '#1a1a1a'] },
          textStyle: { color: '#6b7280' }
        },
        series: [{
          type: 'heatmap',
          data: heatmapData,
          label: { 
            show: true, 
            color: '#fff',
            fontSize: 10,
            formatter: (params) => {
              return params.data ? params.data[2].toFixed(0) : ''
            }
          },
          emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } }
        }]
      })
    }
  })
}

onMounted(() => {
  loadAnalysis()
})
</script>

<style scoped>
.regional-heatmap {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.control-bar {
  display: flex;
  align-items: flex-end;
  gap: 20px;
  padding: 20px 24px;
  background: var(--bg-card);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border);
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;
}

.control-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
  text-align: center;
  transition: all 0.2s ease;
}

.metric-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}

.metric-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-tertiary);
  letter-spacing: 0.05em;
  margin-bottom: 8px;
}

.metric-value {
  font-size: 28px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.02em;
}

.metric-desc {
  font-size: 12px;
  color: var(--text-tertiary);
  margin-top: 4px;
}

.charts-grid {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 24px;
}

.chart-section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}

.section-header h3 {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.tag {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-secondary);
  background: var(--accent-light);
  padding: 4px 10px;
  border-radius: 6px;
}

.chart-container {
  height: 320px;
  width: 100%;
}

.table-section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
}

.priority-tag {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
}

.priority-tag.高 {
  background: #fef2f2;
  color: #dc2626;
}

.priority-tag.中 {
  background: #fffbeb;
  color: #d97706;
}

.priority-tag.低 {
  background: #f3f4f6;
  color: #6b7280;
}

.number {
  font-weight: 600;
}

@media (max-width: 1024px) {
  .charts-grid {
    grid-template-columns: 1fr;
  }
  
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
