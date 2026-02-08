# 搜索引擎提交指南

## 📋 前置准备

### 1. 确认网站已部署
- ✅ 网站地址：https://t.tecfav.com
- ✅ Sitemap：https://t.tecfav.com/sitemap.xml
- ✅ Robots.txt：https://t.tecfav.com/robots.txt

### 2. 验证 Sitemap 可访问
```bash
curl -I https://t.tecfav.com/sitemap.xml
# 应返回 200 OK
```

---

## 🌐 一、Google Search Console

### 步骤 1：注册和验证网站

1. 访问 [Google Search Console](https://search.google.com/search-console)
2. 点击"添加资源"
3. 选择"网址前缀"，输入：`https://t.tecfav.com`

#### 验证方法 A：DNS 验证（推荐）
1. 选择"DNS 记录"验证方式
2. 复制 Google 提供的 TXT 记录
3. 在 Cloudflare DNS 设置中添加：
   ```
   类型: TXT
   名称: @
   内容: google-site-verification=xxxxxxxxxx
   ```
4. 等待几分钟后，点击"验证"

#### 验证方法 B：HTML 标签验证
1. 选择"HTML 标签"验证方式
2. 复制提供的 meta 标签
3. 需要添加到网站 `<head>` 中（告诉我，我会帮你添加）

### 步骤 2：提交 Sitemap

1. 在 Google Search Console 左侧菜单选择"站点地图"
2. 输入：`sitemap.xml`
3. 点击"提交"
4. 状态显示"成功"即可

### 步骤 3：请求索引（可选）

使用"网址检查"工具手动提交重要页面：
```
https://t.tecfav.com/
https://t.tecfav.com/guide/getting-started
https://t.tecfav.com/ai-programming/
https://t.tecfav.com/python/
https://t.tecfav.com/architecture/
```

---

## 🔷 二、Bing Webmaster Tools

### 方法 A：从 Google 导入（最简单）

1. 访问 [Bing Webmaster Tools](https://www.bing.com/webmasters)
2. 使用 Microsoft 账号登录
3. 点击"导入"→"从 Google Search Console 导入"
4. 授权后自动导入网站和 Sitemap

### 方法 B：手动添加

1. 点击"添加站点"
2. 输入：`https://t.tecfav.com`
3. 验证方式：
   - 选择"将 BingSiteAuth.xml 文件放在网站上"
   - 或使用 DNS CNAME 验证

---

## 🔴 三、百度站长平台

### 步骤 1：注册和添加网站

1. 访问 [百度站长平台](https://ziyuan.baidu.com/)
2. 注册/登录百度账号
3. 点击"用户中心"→"站点管理"→"添加网站"
4. 输入：`https://t.tecfav.com`

### 步骤 2：验证网站

#### 方法 A：文件验证
1. 下载验证文件（如 `baidu_verify_xxx.html`）
2. 需要上传到网站根目录（告诉我文件内容，我会帮你添加）

#### 方法 B：HTML 标签验证
1. 选择"HTML 标签验证"
2. 复制提供的 meta 标签（告诉我，我会帮你添加）

#### 方法 C：CNAME 验证（推荐）
1. 选择"CNAME 验证"
2. 在 Cloudflare DNS 添加记录：
   ```
   类型: CNAME
   名称: xxxx（百度提供）
   目标: ziyuan.baidu.com
   ```

### 步骤 3：提交 Sitemap

1. 验证通过后，进入"数据引入"→"链接提交"
2. 选择"sitemap"
3. 输入：`https://t.tecfav.com/sitemap.xml`
4. 点击"提交"

### 步骤 4：手动提交（加速收录）

在"链接提交"→"手动提交"中，粘贴以下 URL：
```
https://t.tecfav.com/
https://t.tecfav.com/guide/getting-started
https://t.tecfav.com/ai-programming/
https://t.tecfav.com/python/
https://t.tecfav.com/go/
https://t.tecfav.com/java/
https://t.tecfav.com/javascript/
https://t.tecfav.com/react/
https://t.tecfav.com/vue/
https://t.tecfav.com/architecture/
https://t.tecfav.com/cloud-native/
https://t.tecfav.com/devops/
https://t.tecfav.com/mysql/
https://t.tecfav.com/redis/
https://t.tecfav.com/datastructures/
```

---

## 🚀 自动化推送（可选）

### 百度主动推送 API

网站已集成自动推送代码，每当有用户访问时自动通知百度。

如需手动批量推送，获取推送 Token 后使用：

```bash
# 从百度站长平台获取 token
TOKEN="your_baidu_token"

# 批量推送
curl -H 'Content-Type:text/plain' \
  --data-binary @docs/public/sitemap-index.txt \
  "http://data.zz.baidu.com/urls?site=t.tecfav.com&token=$TOKEN"
```

### Google Indexing API（需要 API Key）

适用于频繁更新的内容，需要：
1. 在 Google Cloud Console 创建项目
2. 启用 Indexing API
3. 创建服务账号并获取 credentials.json
4. 使用 API 提交 URL

---

## ✅ 验证提交结果

### Google Search Console
- 进入"覆盖率"查看索引状态
- 通常 3-7 天开始收录

### Bing Webmaster Tools
- 查看"索引"→"页面"
- 通常 1-2 周开始收录

### 百度站长平台
- 查看"索引量"曲线
- 通常 1-4 周开始收录

---

## 📊 监控和优化

### 每周检查
- [ ] Google Search Console 覆盖率报告
- [ ] Bing 索引页面数
- [ ] 百度索引量变化

### 优化建议
1. 定期更新内容
2. 在社交媒体分享链接
3. 建立高质量外链
4. 提高页面加载速度
5. 确保移动端友好

---

## ⚡ 快速命令

```bash
# 检查 robots.txt
curl https://t.tecfav.com/robots.txt

# 检查 sitemap.xml
curl https://t.tecfav.com/sitemap.xml | head -20

# 验证网站可访问性
curl -I https://t.tecfav.com
```

---

## 🆘 常见问题

**Q: Sitemap 提交后多久生效？**
A: 通常几小时到几天，不需要重复提交。

**Q: 如何知道页面被收录了？**
A: 在搜索引擎搜索 `site:t.tecfav.com`

**Q: 收录速度慢怎么办？**
A:
1. 确保内容原创、高质量
2. 增加外部链接
3. 保持定期更新
4. 使用手动提交工具

**Q: 需要付费吗？**
A: 不需要，所有搜索引擎的站长工具都是免费的。

---

完成以上步骤后，你的网站将被三大搜索引擎收录并开始获得自然流量！🎉
