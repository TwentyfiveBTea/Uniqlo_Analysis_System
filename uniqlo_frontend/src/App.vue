<template>
  <div class="app-container">
    <!-- 顶部导航 -->
    <header class="top-nav">
      <div class="nav-left">
        <div class="logo-mark">U</div>
        <span class="logo-text">Uniqlo Analytics</span>
      </div>
      <nav class="nav-center">
        <router-link 
          v-for="item in navItems" 
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ active: activeMenu === item.path }"
        >
          <component :is="item.icon" class="nav-icon" />
          <span>{{ item.label }}</span>
        </router-link>
      </nav>
      <div class="nav-right">
        <span class="version-tag">v1.0</span>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="main-content">
      <div class="content-wrapper">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import { TrendCharts, User, MapLocation, Connection } from '@element-plus/icons-vue'

const route = useRoute()
const activeMenu = computed(() => route.path)

const navItems = [
  { path: '/sales-forecast', label: '销售预测', icon: TrendCharts },
  { path: '/user-profiling', label: '用户画像', icon: User },
  { path: '/regional-heatmap', label: '区域分析', icon: MapLocation },
  { path: '/product-association', label: '商品关联', icon: Connection },
]
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

:root {
  --bg-primary: #fafafa;
  --bg-card: #ffffff;
  --text-primary: #1a1a1a;
  --text-secondary: #6b7280;
  --text-tertiary: #9ca3af;
  --accent: #000000;
  --accent-light: #f3f4f6;
  --border: #e5e7eb;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
  --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
  --radius: 12px;
  --radius-lg: 16px;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 顶部导航 - INS/Apple风格 */
.top-nav {
  position: sticky;
  top: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  height: 64px;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
}

.nav-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-mark {
  width: 32px;
  height: 32px;
  background: var(--accent);
  color: white;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 14px;
}

.logo-text {
  font-weight: 500;
  font-size: 15px;
  letter-spacing: -0.02em;
}

.nav-center {
  display: flex;
  gap: 4px;
  background: var(--accent-light);
  padding: 4px;
  border-radius: 10px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: 8px;
  text-decoration: none;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.nav-item:hover {
  color: var(--text-primary);
  background: rgba(255, 255, 255, 0.5);
}

.nav-item.active {
  background: white;
  color: var(--text-primary);
  box-shadow: var(--shadow-sm);
}

.nav-icon {
  width: 16px;
  height: 16px;
}

.nav-right {
  display: flex;
  align-items: center;
}

.version-tag {
  font-size: 11px;
  color: var(--text-tertiary);
  background: var(--accent-light);
  padding: 4px 8px;
  border-radius: 6px;
}

/* 主内容区 */
.main-content {
  flex: 1;
  padding: 32px;
}

.content-wrapper {
  max-width: 1400px;
  margin: 0 auto;
}

/* 页面切换动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* Element Plus 样式覆盖 */
.el-card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
}

.el-card__header {
  border-bottom: 1px solid var(--border);
  padding: 16px 20px;
}

.el-button--primary {
  background: var(--accent);
  border-color: var(--accent);
}

.el-button--primary:hover {
  background: #333;
  border-color: #333;
}

.el-input__wrapper {
  border-radius: 8px;
}

.el-table {
  --el-table-border-color: var(--border);
  --el-table-header-bg-color: var(--bg-card);
}

/* 滚动条美化 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: #d1d5db;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #9ca3af;
}
</style>
