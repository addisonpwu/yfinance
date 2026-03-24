#!/usr/bin/env python3
"""
Create stock_dashboard.html with fully embedded stock.json data
Features:
- All news expanded by default
- Toggle to collapse/expand individual stocks
- Embedded data to avoid CORS issues
"""

import json
from pathlib import Path

# Read the stock.json data
with open('reports/stock.json', 'r', encoding='utf-8') as f:
    stock_data = json.load(f)

# Convert stock data to JSON string for embedding
stock_json_str = json.dumps(stock_data, ensure_ascii=False)

# HTML template with placeholder for embedded data
html_template = '''<!DOCTYPE html>
<html lang="zh-HK">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>港股新聞數據儀表板 | HK Stock News Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #475569;
            --positive: #10b981;
            --neutral: #6b7280;
            --negative: #ef4444;
            --primary: #3b82f6;
            --warning: #f59e0b;
            --info: #06b6d4;
            --purple: #8b5cf6;
            --pink: #ec4899;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        header {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 24px 0;
            margin-bottom: 32px;
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }

        h1 {
            font-size: 1.875rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary) 0%, var(--purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 4px;
        }

        .last-updated {
            color: var(--text-muted);
            font-size: 0.75rem;
            text-align: right;
        }

        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }

        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary), var(--purple));
        }

        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
            border-color: var(--primary);
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .stat-change {
            font-size: 0.875rem;
            margin-top: 8px;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .stat-change.positive { color: var(--positive); }
        .stat-change.negative { color: var(--negative); }

        /* Charts Grid */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 24px;
            margin-bottom: 32px;
        }

        .chart-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            transition: all 0.3s ease;
        }

        .chart-card:hover {
            border-color: var(--border-color);
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .chart-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .chart-subtitle {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
        }

        /* Stock Table */
        .table-section {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 32px;
        }

        .table-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 16px;
        }

        .table-title {
            font-size: 1.25rem;
            font-weight: 600;
        }

        .table-controls {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .search-input {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 16px;
            color: var(--text-primary);
            font-size: 0.875rem;
            width: 250px;
            transition: all 0.2s;
        }

        .search-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .sort-select {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 12px;
            color: var(--text-primary);
            font-size: 0.875rem;
            cursor: pointer;
        }

        .stock-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }

        .stock-table th {
            background: var(--bg-primary);
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 2px solid var(--border-color);
            cursor: pointer;
            transition: all 0.2s;
            user-select: none;
        }

        .stock-table th:hover {
            color: var(--primary);
            background: var(--bg-card);
        }

        .stock-table th.sorted {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }

        .stock-table td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
            transition: all 0.2s;
        }

        .stock-table tbody tr {
            transition: all 0.2s;
        }

        .stock-table tbody tr:hover {
            background: var(--bg-card);
        }

        .stock-table tbody tr.expanded {
            background: var(--bg-card);
        }

        .stock-code {
            font-weight: 600;
            color: var(--primary);
        }

        .stock-name {
            color: var(--text-primary);
        }

        .news-count {
            background: var(--bg-card);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }

        .news-count.high {
            background: rgba(16, 185, 129, 0.2);
            color: var(--positive);
        }

        .news-count.medium {
            background: rgba(245, 158, 11, 0.2);
            color: var(--warning);
        }

        .news-count.low {
            background: rgba(107, 114, 128, 0.2);
            color: var(--text-muted);
        }

        .expand-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            transition: all 0.2s;
            font-size: 0.75rem;
        }

        .expand-btn:hover {
            background: var(--bg-card);
            color: var(--primary);
        }

        /* Expanded Row - Default visible */
        .expanded-row {
            display: table-row;
            background: var(--bg-primary);
        }

        .expanded-row.hide {
            display: none;
        }

        .news-list {
            padding: 20px;
        }

        .news-item {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            transition: all 0.2s;
        }

        .news-item:hover {
            border-color: var(--primary);
            transform: translateX(4px);
        }

        .news-item:last-child {
            margin-bottom: 0;
        }

        .news-meta {
            display: flex;
            gap: 12px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }

        .news-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-rating {
            background: rgba(59, 130, 246, 0.2);
            color: var(--primary);
        }

        .badge-profit {
            background: rgba(16, 185, 129, 0.2);
            color: var(--positive);
        }

        .badge-agency {
            background: var(--bg-card);
            color: var(--text-secondary);
        }

        .news-rating {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-left: 8px;
        }

        .rating-positive {
            background: rgba(16, 185, 129, 0.2);
            color: var(--positive);
        }

        .rating-neutral {
            background: rgba(107, 114, 128, 0.2);
            color: var(--text-muted);
        }

        .rating-negative {
            background: rgba(239, 68, 68, 0.2);
            color: var(--negative);
        }

        .news-title {
            color: var(--text-primary);
            font-weight: 500;
            margin-bottom: 8px;
            line-height: 1.5;
        }

        .news-title a {
            color: inherit;
            text-decoration: none;
            transition: color 0.2s;
        }

        .news-title a:hover {
            color: var(--primary);
        }

        .news-time {
            color: var(--text-muted);
            font-size: 0.75rem;
        }

        /* Sentiment Indicator */
        .sentiment-bar {
            height: 8px;
            background: var(--bg-card);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .sentiment-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--negative) 0%, var(--warning) 50%, var(--positive) 100%);
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }

            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .table-controls {
                flex-direction: column;
                align-items: stretch;
            }

            .search-input {
                width: 100%;
            }

            .stock-table {
                font-size: 0.75rem;
            }

            .stock-table th,
            .stock-table td {
                padding: 8px 12px;
            }
        }

        /* Print Styles */
        @media print {
            body {
                background: white;
                color: black;
            }

            .stat-card,
            .chart-card,
            .table-section {
                break-inside: avoid;
                border: 1px solid #ccc;
            }

            .expand-btn,
            .table-controls {
                display: none;
            }

            .expanded-row {
                display: table-row !important;
            }
        }

        /* Loading State */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 400px;
            font-size: 1.25rem;
            color: var(--text-secondary);
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid var(--border-color);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 16px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Animation */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .chart-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 16px;
            justify-content: center;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div class="header-content">
                <div>
                    <h1>📈 港股新聞數據儀表板</h1>
                    <p class="subtitle">HK Stock News Analytics Dashboard</p>
                </div>
                <div class="last-updated">
                    <div>最後更新 | Last Updated</div>
                    <div id="lastUpdated">-</div>
                </div>
            </div>
        </div>
    </header>

    <main class="container">
        <!-- Stats Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">總股票數量</div>
                <div class="stat-value" id="totalStocks">-</div>
                <div class="stat-change positive">📊 港股主板</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">新聞總數</div>
                <div class="stat-value" id="totalNews">-</div>
                <div class="stat-change" id="avgNewsPerStock">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">正面評級比例</div>
                <div class="stat-value" id="positiveSentiment">-</div>
                <div class="sentiment-bar">
                    <div class="sentiment-fill" id="sentimentFill" style="width: 0%"></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">覆蓋機構數量</div>
                <div class="stat-value" id="totalInstitutions">-</div>
                <div class="stat-change positive">🏦 活躍分析師</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="charts-grid">
            <!-- Chart 1: Rating Distribution -->
            <div class="chart-card">
                <div class="chart-header">
                    <div>
                        <div class="chart-title">評級分佈 | Rating Distribution</div>
                        <div class="chart-subtitle">Doughnut Chart - 各大行評級統計</div>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="ratingChart"></canvas>
                </div>
            </div>

            <!-- Chart 2: News Type -->
            <div class="chart-card">
                <div class="chart-header">
                    <div>
                        <div class="chart-title">新聞類型 | News Type</div>
                        <div class="chart-subtitle">Pie Chart - 評級 vs 業績</div>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="newsTypeChart"></canvas>
                </div>
            </div>

            <!-- Chart 3: Top 10 Stocks -->
            <div class="chart-card">
                <div class="chart-header">
                    <div>
                        <div class="chart-title">最多新聞股票 | Top 10 Stocks</div>
                        <div class="chart-subtitle">Horizontal Bar - 新聞覆蓋數量</div>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="topStocksChart"></canvas>
                </div>
            </div>

            <!-- Chart 4: Top Institutions -->
            <div class="chart-card">
                <div class="chart-header">
                    <div>
                        <div class="chart-title">最活躍機構 | Top Institutions</div>
                        <div class="chart-subtitle">Bar Chart - 機構評級數量</div>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="institutionsChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Stock Table -->
        <div class="table-section">
            <div class="table-header">
                <div>
                    <div class="table-title">股票數據 | Stock Data</div>
                    <div class="chart-subtitle">所有新聞默認展開 | All news expanded by default</div>
                </div>
                <div class="table-controls">
                    <input type="text" class="search-input" id="searchInput" placeholder="搜尋股票代碼/名稱...">
                    <select class="sort-select" id="sortSelect">
                        <option value="newsDesc">新聞數量 ↓</option>
                        <option value="newsAsc">新聞數量 ↑</option>
                        <option value="codeAsc">代碼 A-Z</option>
                        <option value="codeDesc">代碼 Z-A</option>
                        <option value="nameAsc">名稱 A-Z</option>
                        <option value="nameDesc">名稱 Z-A</option>
                    </select>
                </div>
            </div>
            <table class="stock-table">
                <thead>
                    <tr>
                        <th data-sort="code">代碼 <span class="sort-icon"></span></th>
                        <th data-sort="name">名稱 <span class="sort-icon"></span></th>
                        <th data-sort="news">新聞數量 <span class="sort-icon"></span></th>
                        <th>主要評級</th>
                        <th>操作</th>
                    </tr>
                </thead>
                <tbody id="stockTableBody">
                    <!-- Populated by JavaScript -->
                </tbody>
            </table>
        </div>
    </main>

    <script>
        // Global data
        let stockData = [];
        let filteredData = [];
        let charts = {};

        // Rating sentiment mapping
        const ratingSentiment = {
            '強烈推薦': { type: 'positive', score: 3, label: 'Strong Buy' },
            '買入': { type: 'positive', score: 2, label: 'Buy' },
            '增持': { type: 'positive', score: 1, label: 'Overweight' },
            '持有': { type: 'neutral', score: 0, label: 'Hold' },
            '中性': { type: 'neutral', score: 0, label: 'Neutral' },
            '減持': { type: 'negative', score: -1, label: 'Underweight' },
            '沽售': { type: 'negative', score: -2, label: 'Sell' },
            '強烈沽售': { type: 'negative', score: -3, label: 'Strong Sell' },
            '': { type: 'neutral', score: 0, label: 'Unknown' }
        };

        // Color palette
        const colors = {
            positive: '#10b981',
            neutral: '#6b7280',
            negative: '#ef4444',
            primary: '#3b82f6',
            warning: '#f59e0b',
            info: '#06b6d4',
            purple: '#8b5cf6',
            pink: '#ec4899',
            chart: [
                '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
                '#06b6d4', '#ec4899', '#6b7280', '#14b8a6', '#f97316'
            ]
        };

        // Embedded data (to avoid CORS issues when opening file directly)
        const embeddedData = {{EMBEDDED_DATA}};

        // Load and process data
        function loadData() {
            try {
                // Use embedded data directly
                const data = embeddedData;
                stockData = data.stocks || [];
                filteredData = [...stockData];

                // Update last updated time
                if (data.metadata && data.metadata.mergedAt) {
                    const date = new Date(data.metadata.mergedAt);
                    document.getElementById('lastUpdated').textContent =
                        date.toLocaleString('zh-HK', {
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit'
                        });
                }

                processAndRender();
            } catch (error) {
                console.error('Error loading data:', error);
                document.querySelector('.stats-grid').innerHTML =
                    '<div class="loading"><div class="spinner"></div>數據加載失敗 | Data Load Failed</div>';
            }
        }

        // Process data and render all visualizations
        function processAndRender() {
            calculateStats();
            renderRatingChart();
            renderNewsTypeChart();
            renderTopStocksChart();
            renderInstitutionsChart();
            renderTable();
        }

        // Calculate statistics
        function calculateStats() {
            const totalStocks = stockData.length;
            let totalNews = 0;
            let positiveCount = 0;
            let ratingCount = 0;
            const institutions = new Set();

            stockData.forEach(stock => {
                totalNews += stock.news?.length || 0;

                stock.news?.forEach(news => {
                    if (news.type === 'rating' && news.rating) {
                        ratingCount++;
                        const sentiment = ratingSentiment[news.rating] || ratingSentiment[''];
                        if (sentiment.score > 0) {
                            positiveCount++;
                        }
                    }

                    if (news.agency && news.agency.trim()) {
                        institutions.add(news.agency.trim());
                    }
                });
            });

            const positivePct = ratingCount > 0 ? Math.round((positiveCount / ratingCount) * 100) : 0;
            const avgNews = totalStocks > 0 ? (totalNews / totalStocks).toFixed(1) : 0;

            // Update DOM
            document.getElementById('totalStocks').textContent = totalStocks.toLocaleString();
            document.getElementById('totalNews').textContent = totalNews.toLocaleString();
            document.getElementById('positiveSentiment').textContent = `${positivePct}%`;
            document.getElementById('totalInstitutions').textContent = institutions.size.toLocaleString();
            document.getElementById('avgNewsPerStock').textContent = `平均 ${avgNews} 條/股`;
            document.getElementById('sentimentFill').style.width = `${positivePct}%`;
        }

        // Chart 1: Rating Distribution
        function renderRatingChart() {
            const ratingCounts = {};

            stockData.forEach(stock => {
                stock.news?.forEach(news => {
                    if (news.type === 'rating' && news.rating) {
                        ratingCounts[news.rating] = (ratingCounts[news.rating] || 0) + 1;
                    }
                });
            });

            const labels = Object.keys(ratingCounts);
            const data = Object.values(ratingCounts);
            const backgroundColor = labels.map(rating => {
                const sentiment = ratingSentiment[rating]?.type || 'neutral';
                return sentiment === 'positive' ? colors.positive :
                       sentiment === 'negative' ? colors.negative : colors.neutral;
            });

            const ctx = document.getElementById('ratingChart').getContext('2d');
            charts.rating = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: backgroundColor,
                        borderWidth: 2,
                        borderColor: colors.chart[0]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: colors.text,
                                padding: 12,
                                font: { size: 11 }
                            }
                        },
                        tooltip: {
                            backgroundColor: colors.bgSecondary,
                            titleColor: colors.textPrimary,
                            bodyColor: colors.textSecondary,
                            borderColor: colors.border,
                            borderWidth: 1,
                            callbacks: {
                                label: (context) => {
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const pct = ((context.parsed / total) * 100).toFixed(1);
                                    return `${context.label}: ${context.parsed} (${pct}%)`;
                                }
                            }
                        }
                    }
                }
            });
        }

        // Chart 2: News Type Distribution
        function renderNewsTypeChart() {
            const typeCounts = { 'rating': 0, 'profit': 0 };

            stockData.forEach(stock => {
                stock.news?.forEach(news => {
                    if (news.type) {
                        typeCounts[news.type] = (typeCounts[news.type] || 0) + 1;
                    }
                });
            });

            const ctx = document.getElementById('newsTypeChart').getContext('2d');
            charts.newsType = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['評級新聞 | Rating', '業績新聞 | Profit'],
                    datasets: [{
                        data: [typeCounts.rating, typeCounts.profit],
                        backgroundColor: [colors.primary, colors.positive],
                        borderWidth: 2,
                        borderColor: colors.chart[0]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: colors.text,
                                padding: 12,
                                font: { size: 11 }
                            }
                        },
                        tooltip: {
                            backgroundColor: colors.bgSecondary,
                            titleColor: colors.textPrimary,
                            bodyColor: colors.textSecondary,
                            borderColor: colors.border,
                            borderWidth: 1,
                            callbacks: {
                                label: (context) => {
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const pct = ((context.parsed / total) * 100).toFixed(1);
                                    return `${context.label}: ${context.parsed} (${pct}%)`;
                                }
                            }
                        }
                    }
                }
            });
        }

        // Chart 3: Top 10 Stocks by News Count
        function renderTopStocksChart() {
            const stockNewsCount = stockData
                .map(stock => ({
                    code: stock.stockCode,
                    name: stock.stockName,
                    count: stock.news?.length || 0
                }))
                .sort((a, b) => b.count - a.count)
                .slice(0, 10);

            const ctx = document.getElementById('topStocksChart').getContext('2d');
            charts.topStocks = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: stockNewsCount.map(s => `${s.code}\\n${s.name}`),
                    datasets: [{
                        label: '新聞數量',
                        data: stockNewsCount.map(s => s.count),
                        backgroundColor: colors.chart[0],
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: colors.bgSecondary,
                            titleColor: colors.textPrimary,
                            bodyColor: colors.textSecondary,
                            borderColor: colors.border,
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: colors.border },
                            ticks: { color: colors.textSecondary }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { color: colors.textPrimary, font: { size: 10 } }
                        }
                    }
                }
            });
        }

        // Chart 4: Top Institutions
        function renderInstitutionsChart() {
            const institutionCounts = {};

            stockData.forEach(stock => {
                stock.news?.forEach(news => {
                    if (news.agency && news.agency.trim()) {
                        const agency = news.agency.trim();
                        institutionCounts[agency] = (institutionCounts[agency] || 0) + 1;
                    }
                });
            });

            const sorted = Object.entries(institutionCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10);

            const ctx = document.getElementById('institutionsChart').getContext('2d');
            charts.institutions = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: sorted.map(([name]) => name),
                    datasets: [{
                        label: '評級數量',
                        data: sorted.map(([, count]) => count),
                        backgroundColor: colors.chart[3],
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: colors.bgSecondary,
                            titleColor: colors.textPrimary,
                            bodyColor: colors.textSecondary,
                            borderColor: colors.border,
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: {
                                color: colors.textSecondary,
                                maxRotation: 45,
                                minRotation: 45,
                                font: { size: 9 }
                            }
                        },
                        y: {
                            grid: { color: colors.border },
                            ticks: { color: colors.textSecondary }
                        }
                    }
                }
            });
        }

        // Render Stock Table
        function renderTable() {
            const tbody = document.getElementById('stockTableBody');
            tbody.innerHTML = '';

            filteredData.forEach((stock, index) => {
                const newsCount = stock.news?.length || 0;
                const newsClass = newsCount >= 5 ? 'high' : newsCount >= 2 ? 'medium' : 'low';

                // Get primary rating
                let primaryRating = '-';
                let ratingClass = 'neutral';
                if (stock.news?.length > 0) {
                    const ratings = stock.news
                        .filter(n => n.type === 'rating' && n.rating)
                        .map(n => n.rating);
                    if (ratings.length > 0) {
                        primaryRating = ratings[0];
                        const sentiment = ratingSentiment[primaryRating];
                        ratingClass = sentiment?.type || 'neutral';
                    }
                }

                // Main row
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><span class="stock-code">${stock.stockCode}</span></td>
                    <td><span class="stock-name">${stock.stockName}</span></td>
                    <td><span class="news-count ${newsClass}">${newsCount}</span></td>
                    <td>
                        <span class="news-rating rating-${ratingClass}">${primaryRating}</span>
                    </td>
                    <td>
                        <button class="expand-btn" onclick="toggleExpand(${index})">
                            ▼ 收合
                        </button>
                    </td>
                `;
                tbody.appendChild(row);

                // Expanded row (visible by default)
                const expandedRow = document.createElement('tr');
                expandedRow.className = 'expanded-row';
                expandedRow.id = `expanded-${index}`;

                const newsHtml = stock.news?.map(news => {
                    const date = news.publishTime ? new Date(news.publishTime) : null;
                    const timeStr = date ? date.toLocaleString('zh-HK', {
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit'
                    }) : (news.time || 'N/A');

                    const ratingBadge = news.rating ?
                        `<span class="news-rating rating-${ratingSentiment[news.rating]?.type || 'neutral'}">${news.rating}</span>` : '';

                    const agencyBadge = news.agency ?
                        `<span class="news-badge badge-agency">${news.agency}</span>` : '';

                    return `
                        <div class="news-item">
                            <div class="news-meta">
                                <span class="news-badge badge-${news.type || 'rating'}">${news.type === 'profit' ? '業績' : '評級'}</span>
                                ${agencyBadge}
                                ${ratingBadge}
                                <span class="news-time">${timeStr}</span>
                            </div>
                            <div class="news-title">
                                <a href="${news.url || '#'}" target="_blank" rel="noopener">
                                    ${news.title}
                                </a>
                            </div>
                        </div>
                    `;
                }).join('') || '<div style="color: var(--text-muted); padding: 20px; text-align: center;">暫無新聞數據</div>';

                expandedRow.innerHTML = `
                    <td colspan="5">
                        <div class="news-list">
                            ${newsHtml}
                        </div>
                    </td>
                `;
                tbody.appendChild(expandedRow);
            });
        }

        // Toggle expand row
        function toggleExpand(index) {
            const expandedRow = document.getElementById(`expanded-${index}`);
            const row = expandedRow.previousElementSibling;
            const btn = row.querySelector('.expand-btn');

            if (expandedRow.classList.contains('hide')) {
                // Show
                expandedRow.classList.remove('hide');
                row.classList.add('expanded');
                btn.textContent = '▼ 收合';
            } else {
                // Hide
                expandedRow.classList.add('hide');
                row.classList.remove('expanded');
                btn.textContent = '▶ 展開';
            }
        }

        // Search and filter
        document.getElementById('searchInput').addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase().trim();
            filteredData = stockData.filter(stock =>
                stock.stockCode.toLowerCase().includes(term) ||
                stock.stockName.toLowerCase().includes(term)
            );
            applySort();
        });

        // Sort
        document.getElementById('sortSelect').addEventListener('change', applySort);

        function applySort() {
            const sortValue = document.getElementById('sortSelect').value;

            filteredData.sort((a, b) => {
                const aNews = a.news?.length || 0;
                const bNews = b.news?.length || 0;

                switch(sortValue) {
                    case 'newsDesc': return bNews - aNews;
                    case 'newsAsc': return aNews - bNews;
                    case 'codeAsc': return a.stockCode.localeCompare(b.stockCode);
                    case 'codeDesc': return b.stockCode.localeCompare(a.stockCode);
                    case 'nameAsc': return a.stockName.localeCompare(b.stockName, 'zh-HK');
                    case 'nameDesc': return b.stockName.localeCompare(a.stockName, 'zh-HK');
                    default: return 0;
                }
            });

            renderTable();
        }

        // Table header sort
        document.querySelectorAll('.stock-table th[data-sort]').forEach(th => {
            th.addEventListener('click', () => {
                const sortField = th.dataset.sort;
                const select = document.getElementById('sortSelect');

                // Map field to select value
                const sortMap = {
                    'code': 'codeAsc',
                    'name': 'nameAsc',
                    'news': 'newsDesc'
                };

                select.value = sortMap[sortField];
                applySort();

                // Update sorted class
                document.querySelectorAll('.stock-table th').forEach(h => h.classList.remove('sorted'));
                th.classList.add('sorted');
            });
        });

        // Initialize
        document.addEventListener('DOMContentLoaded', loadData);
    </script>
</body>
</html>
'''

# Replace placeholder with actual data
html_output = html_template.replace('{{EMBEDDED_DATA}}', stock_json_str)

# Write the updated HTML
with open('reports/stock_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

print("✅ Successfully created stock_dashboard.html with embedded data")
print(f"📊 Total stocks: {len(stock_data.get('stocks', []))}")
print(f"📰 Total news items: {sum(len(s.get('news', [])) for s in stock_data.get('stocks', []))}")
print(f"📁 File size: {Path('reports/stock_dashboard.html').stat().st_size / 1024:.1f} KB")
print(f"✨ Feature: All news expanded by default")
