<template>
  <div class="product-association">
    <!-- 控制栏 -->
    <div class="control-bar">
      <div class="control-group">
        <label class="control-label">最小支持度: {{ (minSupport * 100).toFixed(1) }}%</label>
        <el-slider v-model="minSupport" :min="0.001" :max="0.1" :step="0.001" @change="loadAnalysis" />
      </div>
      <div class="control-group">
        <label class="control-label">最小置信度: {{ (minConfidence * 100).toFixed(0) }}%</label>
        <el-slider v-model="minConfidence" :min="0.05" :max="0.5" :step="0.05" @change="loadAnalysis" />
      </div>
      <el-button type="primary" size="large" @click="loadAnalysis" :loading="loading">
        挖掘规则
      </el-button>
    </div>

    <!-- 指标卡片 -->
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">交易总数</div>
        <div class="metric-value">{{ totalTransactions?.toLocaleString() || '-' }}</div>
        <div class="metric-desc">有效交易记录</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">频繁项集</div>
        <div class="metric-value">{{ frequentItemsets?.length || 0 }}</div>
        <div class="metric-desc">发现的频繁项集</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">关联规则</div>
        <div class="metric-value">{{ associationRules?.length || 0 }}</div>
        <div class="metric-desc">挖掘的关联规则</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">执行时间</div>
        <div class="metric-value">{{ executionTime?.toFixed(2) || '-' }}s</div>
        <div class="metric-desc">算法运行耗时</div>
      </div>
    </div>

    <!-- 图表区域 -->
    <div class="charts-grid">
      <div class="chart-section">
        <div class="section-header">
          <h3>频繁项集支持度</h3>
          <span class="tag">Apriori</span>
        </div>
        <div ref="itemsetChartRef" class="chart-container"></div>
      </div>
      <div class="chart-section">
        <div class="section-header">
          <h3>关联规则网络图</h3>
        </div>
        <div ref="networkChartRef" class="chart-container"></div>
      </div>
    </div>

    <!-- 关联规则表格 -->
    <div class="table-section">
      <div class="section-header">
        <h3>关联规则详情</h3>
      </div>
      <el-table :data="associationRules" stripe size="large">
        <el-table-column label="规则" min-width="280">
          <template #default="{ row }">
            <span class="rule-text">
              {{ row.antecedent?.join(' + ') }} <span class="arrow">→</span> {{ row.consequent?.join(' + ') }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="支持度" width="100">
          <template #default="{ row }">
            <span class="metric-val">{{ (row.support * 100).toFixed(2) }}%</span>
          </template>
        </el-table-column>
        <el-table-column label="置信度" width="100">
          <template #default="{ row }">
            <span class="metric-val">{{ (row.confidence * 100).toFixed(2) }}%</span>
          </template>
        </el-table-column>
        <el-table-column label="提升度" width="100">
          <template #default="{ row }">
            <span class="metric-val">{{ row.lift?.toFixed(3) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="价值" width="100">
          <template #default="{ row }">
            <span class="value-tag" :class="row.lift > 1.5 ? 'high' : row.lift > 1 ? 'mid' : 'low'">
              {{ row.lift > 1.5 ? '高价值' : row.lift > 1 ? '中等' : '一般' }}
            </span>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'

const loading = ref(false)
const minSupport = ref(0.005)
const minConfidence = ref(0.1)
const frequentItemsets = ref([])
const associationRules = ref([])
const totalTransactions = ref(0)
const executionTime = ref(0)
const itemsetChartRef = ref(null)
const networkChartRef = ref(null)
let itemsetChart = null
let networkChart = null

const loadAnalysis = async () => {
  loading.value = true
  try {
    const res = await axios.get('/api/apriori/analysis', {
      params: {
        min_support: minSupport.value,
        min_confidence: minConfidence.value
      }
    })
    const data = res.data
    
    frequentItemsets.value = data.frequent_itemsets || []
    associationRules.value = data.association_rules || []
    totalTransactions.value = data.total_transactions || 0
    executionTime.value = data.execution_time || 0
    
    updateCharts()
  } catch (e) {
    console.error('Failed to load analysis:', e)
  } finally {
    loading.value = false
  }
}

const updateCharts = () => {
  nextTick(() => {
    if (itemsetChartRef.value) {
      if (itemsetChart) itemsetChart.dispose()
      itemsetChart = echarts.init(itemsetChartRef.value)
      
      const topItemsets = frequentItemsets.value.slice(0, 10)
      
      itemsetChart.setOption({
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#fff', borderColor: '#e5e7eb', textStyle: { color: '#1a1a1a' } },
        grid: { left: '3%', right: '15%', bottom: '3%', containLabel: true },
        xAxis: { type: 'value', axisLabel: { color: '#6b7280' } },
        yAxis: { 
          type: 'category', 
          data: topItemsets.map(i => i.items?.join(' + ')),
          axisLabel: { color: '#1a1a1a' }
        },
        series: [{
          type: 'bar',
          data: topItemsets.map(i => (i.support * 100).toFixed(2)),
          itemStyle: { color: '#1a1a1a' },
          label: {
            show: true,
            position: 'right',
            formatter: '{c}%',
            color: '#6b7280'
          }
        }]
      })
    }
    
    if (networkChartRef.value) {
      if (networkChart) networkChart.dispose()
      networkChart = echarts.init(networkChartRef.value)
      
      const allItems = new Set()
      associationRules.value.slice(0, 15).forEach(rule => {
        rule.antecedent?.forEach(item => allItems.add(item))
        rule.consequent?.forEach(item => allItems.add(item))
      })
      
      const nodes = Array.from(allItems).map(item => ({
        name: item,
        symbolSize: 25 + Math.random() * 15,
        value: 1,
        category: 0
      }))
      
      const links = associationRules.value.slice(0, 15).map(rule => ({
        source: rule.antecedent?.[0] || '',
        target: rule.consequent?.[0] || '',
        value: rule.lift,
        lineStyle: {
          width: Math.max(rule.lift, 1),
          opacity: 0.6
        }
      }))
      
      networkChart.setOption({
        tooltip: { backgroundColor: '#fff', borderColor: '#e5e7eb', textStyle: { color: '#1a1a1a' } },
        series: [{
          type: 'graph',
          layout: 'force',
          data: nodes,
          links: links,
          roam: true,
          label: { show: true, position: 'right', color: '#1a1a1a' },
          force: {
            repulsion: 150,
            edgeLength: 80
          },
          lineStyle: {
            color: '#6b7280',
            curveness: 0.1
          }
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
.product-association {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.control-bar {
  display: flex;
  align-items: flex-end;
  gap: 24px;
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

.rule-text {
  font-weight: 500;
  color: var(--text-primary);
}

.arrow {
  color: var(--text-tertiary);
  margin: 0 8px;
}

.metric-val {
  font-weight: 600;
  color: var(--text-primary);
}

.value-tag {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
}

.value-tag.high {
  background: #f0fdf4;
  color: #16a34a;
}

.value-tag.mid {
  background: #fffbeb;
  color: #d97706;
}

.value-tag.low {
  background: #f3f4f6;
  color: #6b7280;
}

@media (max-width: 1024px) {
  .charts-grid {
    grid-template-columns: 1fr;
  }
  
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .control-bar {
    flex-direction: column;
  }
}
</style>
