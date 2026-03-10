import { createRouter, createWebHistory } from 'vue-router'
import SalesForecast from './views/SalesForecast.vue'
import UserProfiling from './views/UserProfiling.vue'
import RegionalHeatmap from './views/RegionalHeatmap.vue'
import ProductAssociation from './views/ProductAssociation.vue'

const routes = [
  {
    path: '/',
    redirect: '/sales-forecast'
  },
  {
    path: '/sales-forecast',
    name: 'SalesForecast',
    component: SalesForecast,
    meta: { title: '销售趋势预测' }
  },
  {
    path: '/user-profiling',
    name: 'UserProfiling',
    component: UserProfiling,
    meta: { title: '用户画像分析' }
  },
  {
    path: '/regional-heatmap',
    name: 'RegionalHeatmap',
    component: RegionalHeatmap,
    meta: { title: '区域销售分析' }
  },
  {
    path: '/product-association',
    name: 'ProductAssociation',
    component: ProductAssociation,
    meta: { title: '商品关联分析' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
