<template>
  <div class="sales-forecast">
    <!-- 简洁控制栏 -->
    <div class="control-bar">
      <div class="control-group">
        <label class="control-label">品类</label>
        <el-select v-model="selectedCategory" size="large" @change="loadForecast">
          <el-option v-for="cat in categories" :key="cat.id" :label="cat.name" :value="cat.id" />
        </el-select>
      </div>
      <div class="control-group">
        <label class="control-label">预测天数</label>
        <el-input-number v-model="forecastDays" :min="7" :max="90" size="large" @change="loadForecast" />
      </div>
      <el-button type="primary" size="large" @click="loadForecast" :loading="loading">
        生成预测
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

    <!-- 图表 -->
    <div class="chart-section">
      <div class="section-header">
        <h3>销售趋势预测</h3>
        <span class="tag">ARIMA</span>
      </div>
      <div ref="chartRef" class="chart-container"></div>
    </div>

    <!-- 预测表格 -->
    <div class="table-section">
      <div class="section-header">
        <h3>未来 {{ forecastDays }} 天销量预测</h3>
      </div>
      <el-table :data="forecastData" stripe size="large">
        <el-table-column prop="date" label="日期" width="140" />
        <el-table-column label="预测销量" width="160">
          <template #default="{ row }">
            <span class="forecast-value">{{ row.value?.toFixed(0) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="置信区间">
          <template #default="{ row, $index }">
            <span class="confidence-range">
              {{ confidenceInterval[$index]?.lower?.toFixed(0) }} - {{ confidenceInterval[$index]?.upper?.toFixed(0) }}
            </span>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'

const loading = ref(false)
const selectedCategory = ref('')
const forecastDays = ref(30)
const categories = ref([])
const forecastData = ref([])
const historicalData = ref([])
const confidenceInterval = ref([])
const metrics = ref({})
const chartRef = ref(null)
let chart = null

const metricItems = computed(() => [
  { key: 'rmse', label: 'RMSE', value: metrics.value.rmse?.toFixed(2) || '-', desc: '均方根误差' },
  { key: 'mae', label: 'MAE', value: metrics.value.mae?.toFixed(2) || '-', desc: '平均绝对误差' },
  { key: 'mape', label: 'MAPE', value: (metrics.value.mape?.toFixed(2) || '-') + '%', desc: '百分比误差' },
  { key: 'aic', label: 'AIC', value: metrics.value.aic?.toFixed(0) || '-', desc: '模型复杂度' },
])

const loadCategories = async () => {
  try {
    const res = await axios.get('/api/arima/categories')
    categories.value = res.data
    if (categories.value.length > 0) {
      selectedCategory.value = categories.value[0].id
    }
  } catch (e) {
    console.error('Failed to load categories:', e)
  }
}

const loadForecast = async () => {
  loading.value = true
  try {
    const res = await axios.get('/api/arima/forecast', {
      params: {
        category: selectedCategory.value,
        forecast_days: forecastDays.value
      }
    })
    const data = res.data
    historicalData.value = data.historical_data || []
    forecastData.value = data.forecast_data || []
    confidenceInterval.value = data.confidence_interval || []
    metrics.value = data.metrics || {}
    updateChart()
  } catch (e) {
    console.error('Failed to load forecast:', e)
  } finally {
    loading.value = false
  }
}

const updateChart = () => {
  if (!chartRef.value) return
  
  nextTick(() => {
    if (chart) {
      chart.dispose()
    }
    
    chart = echarts.init(chartRef.value)
    
    const historyDates = historicalData.value.slice(-60).map(d => d.date)
    const historyValues = historicalData.value.slice(-60).map(d => d.value)
    const forecastDates = forecastData.value.map(d => d.date)
    const forecastValues = forecastData.value.map(d => d.value)
    const upperValues = confidenceInterval.value.map(d => d.upper)
    const lowerValues = confidenceInterval.value.map(d => d.lower)
    
    const option = {
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#fff',
        borderColor: '#e5e7eb',
        textStyle: { color: '#1a1a1a' }
      },
      legend: {
        data: ['历史销量', '预测销量', '置信区间'],
        bottom: 0
      },
      grid: { left: '3%', right: '4%', bottom: '15%', top: '10%', containLabel: true },
      xAxis: {
        type: 'category',
        data: [...historyDates, ...forecastDates],
        axisLine: { lineStyle: { color: '#e5e7eb' } },
        axisLabel: { color: '#6b7280', rotate: 45 }
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisLabel: { color: '#6b7280' },
        splitLine: { lineStyle: { color: '#f3f4f6' } }
      },
      series: [
        {
          name: '历史销量',
          type: 'line',
          data: [...historyValues, ...new Array(forecastDays.value).fill(null)],
          smooth: true,
          lineStyle: { color: '#9ca3af', width: 2 },
          itemStyle: { color: '#9ca3af' }
        },
        {
          name: '预测销量',
          type: 'line',
          data: [...new Array(historyDates.length - 1).fill(null), historyValues[historyValues.length - 1], ...forecastValues],
          smooth: true,
          lineStyle: { color: '#000', width: 2.5 },
          itemStyle: { color: '#000' }
        },
        {
          name: '置信区间',
          type: 'line',
          data: [...new Array(historyDates.length - 1).fill(null), historyValues[historyValues.length - 1], ...upperValues],
          smooth: true,
          lineStyle: { type: 'dashed', color: '#000', opacity: 0.3 },
          areaStyle: { color: 'rgba(0,0,0,0.05)' }
        }
      ]
    }
    
    chart.setOption(option)
  })
}

onMounted(async () => {
  await loadCategories()
  await loadForecast()
})
</script>

<style scoped>
.sales-forecast {
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

.chart-section, .table-section {
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
  letter-spacing: 0.02em;
}

.chart-container {
  height: 380px;
  width: 100%;
}

.forecast-value {
  font-weight: 600;
  color: var(--text-primary);
}

.confidence-range {
  color: var(--text-secondary);
  font-size: 13px;
}

@media (max-width: 768px) {
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .control-bar {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
