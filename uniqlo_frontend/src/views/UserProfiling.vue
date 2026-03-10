<template>
  <div class="user-profiling">
    <!-- 控制栏 -->
    <div class="control-bar">
      <div class="control-group">
        <label class="control-label">聚类数量</label>
        <el-input-number v-model="nClusters" :min="3" :max="10" size="large" @change="loadClustering" />
      </div>
      <el-button type="primary" size="large" @click="loadClustering" :loading="loading">
        重新聚类
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
          <h3>用户群体分布</h3>
          <span class="tag">K-means</span>
        </div>
        <div ref="pieChartRef" class="chart-container"></div>
      </div>
      <div class="chart-section">
        <div class="section-header">
          <h3>各群体特征对比</h3>
        </div>
        <div ref="radarChartRef" class="chart-container"></div>
      </div>
    </div>

    <!-- 用户群体详情 -->
    <div class="table-section">
      <div class="section-header">
        <h3>用户群体详情</h3>
      </div>
      <el-table :data="clusters" stripe size="large">
        <el-table-column prop="portrait_label" label="群体标签" width="140" />
        <el-table-column prop="user_count" label="用户数" width="100" />
        <el-table-column prop="avg_total_orders" label="平均订单数" width="120" />
        <el-table-column label="平均消费(元)" width="140">
          <template #default="{ row }">
            <span class="money">¥{{ row.avg_total_spend?.toFixed(2) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="avg_purchase_frequency" label="购买频次/月" width="130" />
        <el-table-column prop="top_categories" label="偏好品类" width="180" />
        <el-table-column prop="description" label="群体描述" />
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'

const loading = ref(false)
const nClusters = ref(4) // 默认4个聚类
const clusters = ref([])
const metrics = ref({})
const optimalK = ref(0)
const pieChartRef = ref(null)
const radarChartRef = ref(null)
let pieChart = null
let radarChart = null

const metricItems = computed(() => [
  { key: 'silhouette', label: '轮廓系数', value: metrics.value.silhouette_score?.toFixed(3) || '-', desc: '聚类质量' },
  { key: 'calinski', label: 'CH指数', value: metrics.value.calinski_harabasz_score?.toFixed(2) || '-', desc: '簇间分离度' },
  { key: 'davies', label: 'DB指数', value: metrics.value.davies_bouldin_score?.toFixed(3) || '-', desc: '簇内紧凑度' },
  { key: 'optimalK', label: '最优K值', value: optimalK.value || '-', desc: '推荐聚类数' },
])

const loadClustering = async () => {
  loading.value = true
  try {
    const params = nClusters.value > 0 ? { n_clusters: nClusters.value } : {}
    const res = await axios.get('/api/kmeans/clustering', { params })
    const data = res.data
    
    console.log('API返回数据:', JSON.stringify(data, null, 2))
    console.log('clusters数据:', JSON.stringify(data.clusters, null, 2))
    
    clusters.value = data.clusters || []
    metrics.value = data.metrics || {}
    optimalK.value = data.optimal_k || 0
    
    updateCharts()
  } catch (e) {
    console.error('Failed to load clustering:', e)
  } finally {
    loading.value = false
  }
}

const updateCharts = () => {
  nextTick(() => {
    if (pieChartRef.value) {
      if (pieChart) pieChart.dispose()
      pieChart = echarts.init(pieChartRef.value)
      
      const colors = ['#000000', '#6b7280', '#9ca3af', '#d1d5db', '#f3f4f6']
      const pieData = clusters.value.map((c, i) => ({
        name: c.portrait_label,
        value: c.user_count,
        itemStyle: { color: colors[i % colors.length] }
      }))
      
      pieChart.setOption({
        tooltip: { trigger: 'item' },
        legend: { bottom: '0%' },
        series: [{
          type: 'pie',
          radius: ['40%', '70%'],
          data: pieData,
          label: { show: false }
        }]
      })
    }
    
    if (radarChartRef.value) {
      if (radarChart) radarChart.dispose()
      radarChart = echarts.init(radarChartRef.value)
      
      // 使用动态最大值，让图表更准确地反映数据差异
      const getMax = (arr, key) => Math.max(...arr.map(c => c[key] || 0)) * 1.2
      
      const ordersMax = Math.max(getMax(clusters.value, 'avg_total_orders'), 10)
      const spendMax = Math.max(getMax(clusters.value, 'avg_total_spend'), 5000)
      const orderValueMax = Math.max(getMax(clusters.value, 'avg_order_value'), 500)
      const freqMax = Math.max(getMax(clusters.value, 'avg_purchase_frequency'), 1)
      const daysMax = Math.max(getMax(clusters.value, 'avg_days_since_last_purchase'), 50)
      
      const indicators = [
        { name: '订单数', max: ordersMax },
        { name: '消费金额', max: spendMax },
        { name: '客单价', max: orderValueMax },
        { name: '购买频次', max: freqMax },
        { name: '最近购买(天)', max: daysMax }
      ]
      
      const radarData = clusters.value.map(c => ({
        value: [
          c.avg_total_orders || 0,
          c.avg_total_spend || 0,
          c.avg_order_value || 0,
          c.avg_purchase_frequency || 0,
          c.avg_days_since_last_purchase || 0
        ],
        name: c.portrait_label
      }))
      
      console.log('雷达图indicators:', indicators)
      console.log('雷达图radarData:', radarData)
      
      radarChart.setOption({
        tooltip: {},
        legend: { bottom: '0%', data: clusters.value.map(c => c.portrait_label) },
        radar: { 
          indicator: indicators,
          axisName: { color: '#6b7280' }
        },
        series: [{
          type: 'radar',
          data: radarData,
          lineStyle: { color: '#000' },
          areaStyle: { color: 'rgba(0,0,0,0.1)' }
        }]
      })
    }
  })
}

onMounted(() => {
  loadClustering()
})
</script>

<style scoped>
.user-profiling {
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

.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
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

.money {
  font-weight: 600;
  color: var(--text-primary);
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
