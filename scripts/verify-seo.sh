#!/bin/bash

# SEO 配置验证脚本

echo "🔍 验证 SEO 配置..."
echo "================================"

# 网站基本信息
SITE="https://t.tecfav.com"

echo ""
echo "📍 网站信息"
echo "  网站地址: $SITE"
echo ""

# 检查网站可访问性
echo "1️⃣ 检查网站可访问性..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $SITE)
if [ "$STATUS" = "200" ]; then
    echo "  ✅ 网站可访问 (HTTP $STATUS)"
else
    echo "  ❌ 网站不可访问 (HTTP $STATUS)"
fi
echo ""

# 检查 robots.txt
echo "2️⃣ 检查 robots.txt..."
ROBOTS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $SITE/robots.txt)
if [ "$ROBOTS_STATUS" = "200" ]; then
    echo "  ✅ robots.txt 存在"
    echo "  📄 内容预览:"
    curl -s $SITE/robots.txt | head -10 | sed 's/^/     /'
else
    echo "  ❌ robots.txt 不存在"
fi
echo ""

# 检查 sitemap.xml
echo "3️⃣ 检查 sitemap.xml..."
SITEMAP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $SITE/sitemap.xml)
if [ "$SITEMAP_STATUS" = "200" ]; then
    echo "  ✅ sitemap.xml 存在"
    URL_COUNT=$(curl -s $SITE/sitemap.xml | grep -c "<loc>")
    echo "  📊 包含 $URL_COUNT 个 URL"
else
    echo "  ❌ sitemap.xml 不存在"
fi
echo ""

# 检查关键页面
echo "4️⃣ 检查关键页面..."
PAGES=(
    "/"
    "/guide/getting-started"
    "/python/"
    "/ai-programming/"
    "/architecture/"
)

for page in "${PAGES[@]}"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$SITE$page")
    if [ "$STATUS" = "200" ]; then
        echo "  ✅ $page"
    else
        echo "  ❌ $page (HTTP $STATUS)"
    fi
done
echo ""

# 检查 Meta 标签
echo "5️⃣ 检查 Meta 标签（首页）..."
HTML=$(curl -s $SITE)

if echo "$HTML" | grep -q "og:title"; then
    echo "  ✅ Open Graph 标签"
else
    echo "  ❌ 缺少 Open Graph 标签"
fi

if echo "$HTML" | grep -q "twitter:card"; then
    echo "  ✅ Twitter Card 标签"
else
    echo "  ❌ 缺少 Twitter Card 标签"
fi

if echo "$HTML" | grep -q "application/ld+json"; then
    echo "  ✅ 结构化数据 (JSON-LD)"
else
    echo "  ❌ 缺少结构化数据"
fi

if echo "$HTML" | grep -q "gtag"; then
    echo "  ✅ Google Analytics"
else
    echo "  ❌ 缺少 Google Analytics"
fi
echo ""

# 搜索引擎收录检查
echo "6️⃣ 搜索引擎收录检查..."
echo "  💡 在浏览器中手动检查："
echo "     Google: site:t.tecfav.com"
echo "     Bing:   site:t.tecfav.com"
echo "     百度:   site:t.tecfav.com"
echo ""

# 下一步提示
echo "================================"
echo "📋 下一步操作："
echo ""
echo "1. Google Search Console"
echo "   👉 https://search.google.com/search-console"
echo "   - 添加资源并验证"
echo "   - 提交 sitemap.xml"
echo ""
echo "2. Bing Webmaster Tools"
echo "   👉 https://www.bing.com/webmasters"
echo "   - 从 Google 导入或手动添加"
echo ""
echo "3. 百度站长平台"
echo "   👉 https://ziyuan.baidu.com/"
echo "   - 添加网站并验证"
echo "   - 提交 sitemap 和 URL"
echo ""
echo "详细步骤请查看: scripts/submit-to-search-engines.md"
echo "================================"
