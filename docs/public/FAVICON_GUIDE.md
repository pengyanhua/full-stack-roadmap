# Favicon 指南

## 当前配置

项目使用 SVG favicon，支持自动适配深浅色模式。

### 文件位置

- `docs/public/favicon.svg` - SVG 格式 favicon（推荐）
- `docs/public/logo.svg` - 网站 logo（也用作备用 favicon）

### VitePress 配置

在 `docs/.vitepress/config.ts` 中已配置：

```typescript
head: [
  ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
  ['link', { rel: 'icon', type: 'image/png', href: '/logo.svg' }],
  // ... 其他 meta 标签
]
```

## 自定义 Favicon

### 方法 1：替换 SVG 文件（推荐）

直接编辑 `docs/public/favicon.svg`：

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <!-- 你的设计 -->
</svg>
```

**优势：**
- 矢量图，任何尺寸都清晰
- 文件小
- 支持自定义颜色

### 方法 2：使用 PNG/ICO

1. 准备多个尺寸的图标：
   - 16x16 (浏览器标签)
   - 32x32 (浏览器标签)
   - 48x48 (Windows 任务栏)
   - 180x180 (iOS)
   - 192x192 (Android)
   - 512x512 (PWA)

2. 使用在线工具生成：
   - [RealFaviconGenerator](https://realfavicongenerator.net/)
   - [Favicon.io](https://favicon.io/)

3. 将生成的文件放到 `docs/public/`

4. 更新配置：

```typescript
head: [
  ['link', { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }],
  ['link', { rel: 'apple-touch-icon', sizes: '180x180', href: '/apple-touch-icon.png' }],
  ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon-32x32.png' }],
  ['link', { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/favicon-16x16.png' }]
]
```

## 在线生成工具

### RealFaviconGenerator（推荐）

1. 访问：https://realfavicongenerator.net/
2. 上传你的 logo（至少 512x512）
3. 自定义各平台的显示效果
4. 下载生成的包
5. 将文件复制到 `docs/public/`
6. 更新 `config.ts` 中的配置

### Favicon.io

1. 访问：https://favicon.io/
2. 选择方式：
   - 从文本生成
   - 从图片生成
   - 从 emoji 生成
3. 下载并使用

## 测试 Favicon

### 本地测试

```bash
npm run docs:dev
```

打开浏览器，查看标签页图标。

### 在线测试

部署后访问：
- Chrome/Edge: 标签页左上角
- Safari: 标签页
- 移动端: 添加到主屏幕后的图标

### 验证工具

- [Favicon Checker](https://realfavicongenerator.net/favicon_checker)
- 浏览器开发者工具 → Network → 查看 favicon 请求

## 最佳实践

### SVG Favicon（现代浏览器）

✅ **优点：**
- 矢量图，永远清晰
- 文件小（几 KB）
- 可以内联 CSS 实现主题切换

✅ **支持：**
- Chrome 80+
- Firefox 41+
- Safari 9+
- Edge 79+

### 传统 ICO（兼容性）

✅ **优点：**
- 兼容所有浏览器
- 包含多个尺寸

⚠️ **缺点：**
- 文件较大
- 不支持透明度动画

## 当前设计说明

现有 `favicon.svg` 使用：
- 渐变色背景（紫蓝色）
- 代码符号 `</>`
- 简洁现代的设计

如需修改：
1. 保持 100x100 视口
2. 使用高对比度颜色
3. 避免过于复杂的细节（16x16 时不清晰）

## PWA 配置（可选）

如果未来需要 PWA 支持，添加 manifest：

```json
// docs/public/manifest.json
{
  "name": "全栈开发学习路线",
  "short_name": "全栈路线",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "theme_color": "#4F46E5",
  "background_color": "#ffffff",
  "display": "standalone"
}
```

然后在 config.ts 中引用：

```typescript
['link', { rel: 'manifest', href: '/manifest.json' }]
```

---

**当前配置已满足大多数需求。** 如果需要更专业的 favicon，建议使用 RealFaviconGenerator 生成完整的图标包。
