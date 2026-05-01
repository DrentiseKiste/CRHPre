"""
中国高铁历史线路数据爬虫
==========================
用途：爬取中国高铁各年份开通线路数据，用于构建时间序列数据集，
      支持动态可视化展示高铁网络从2008年至今的扩展过程。

数据来源分析：
---------------
1. **Wikipedia (推荐)**：
    - 页面："中国高速铁路"、"中国高铁线路列表"
    - 优点：结构化程度高，有完整的开通日期表格
    - URL模式：https://zh.wikipedia.org/wiki/中国高速铁路线路列表

2. **国家铁路局/中国铁路官网**：
    - 有官方开通公告，但分散在新闻稿中，结构化程度低

3. **第三方聚合网站**（如铁道建设规划、人民铁道等）：
    - 有年度汇总文章，但格式不统一

爬取策略：
----------
由于 Wikipedia 的数据最结构化且最完整，本脚本主要基于 Wikipedia 的
"中国高速铁路线路列表"页面进行爬取。同时提供备选方案：
手动整理的关键年份数据（当爬虫不可用时作为fallback）。

数据输出格式：
--------------
CSV 文件，包含字段：
    - line_name: 线路名称
    - start_city: 起点城市
    - end_city: 终点城市
    - open_year: 开通年份
    - open_date: 开通日期（YYYY-MM-DD，如有）
    - length_km: 线路长度（公里，如有）
    - max_speed: 设计最高时速（km/h，如有）
    - source: 数据来源

使用说明：
----------
1. 直接运行本脚本：python crawler_hsr_history.py
2. 若 Wikipedia 访问受限，脚本会自动使用内置的 fallback 数据
3. 输出的 CSV 文件可直接用于主脚本的时间序列分析

注意事项：
----------
- Wikipedia 页面结构可能变化，若解析失败请检查页面HTML结构
- 部分线路分段开通（如京广高铁），本脚本会记录各段独立的开通时间
- 建议爬取后人工核对关键线路（如京沪高铁、京广高铁等）的开通日期
"""

import csv
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[警告] requests 或 beautifulsoup4 未安装，将使用内置数据。")
    print("  安装命令: pip install requests beautifulsoup4 lxml")


# =============================================================================
# Fallback 数据：中国高铁关键线路开通历史（当爬虫不可用时使用）
# =============================================================================
# 数据来源：Wikipedia、国家铁路局公告、新华社报道等公开资料
# 整理原则：记录各线路（或分段）的首次开通年份
FALLBACK_HSR_HISTORY: List[Dict[str, Any]] = [
    # 2008年
    {"line_name": "京津城际铁路", "start_city": "北京", "end_city": "天津", "open_year": 2008, "open_date": "2008-08-01", "length_km": 120, "max_speed": 350},
    # 2009年
    {"line_name": "武广高速铁路", "start_city": "武汉", "end_city": "广州", "open_year": 2009, "open_date": "2009-12-26", "length_km": 968, "max_speed": 350},
    {"line_name": "石太客运专线", "start_city": "石家庄", "end_city": "太原", "open_year": 2009, "open_date": "2009-04-01", "length_km": 225, "max_speed": 250},
    {"line_name": "合武铁路", "start_city": "合肥", "end_city": "武汉", "open_year": 2009, "open_date": "2009-04-01", "length_km": 359, "max_speed": 250},
    {"line_name": "甬台温铁路", "start_city": "宁波", "end_city": "温州", "open_year": 2009, "open_date": "2009-09-28", "length_km": 268, "max_speed": 250},
    {"line_name": "温福铁路", "start_city": "温州", "end_city": "福州", "open_year": 2009, "open_date": "2009-09-28", "length_km": 298, "max_speed": 250},
    # 2010年
    {"line_name": "郑西高速铁路", "start_city": "郑州", "end_city": "西安", "open_year": 2010, "open_date": "2010-02-06", "length_km": 457, "max_speed": 350},
    {"line_name": "沪宁城际铁路", "start_city": "上海", "end_city": "南京", "open_year": 2010, "open_date": "2010-07-01", "length_km": 301, "max_speed": 350},
    {"line_name": "沪杭高速铁路", "start_city": "上海", "end_city": "杭州", "open_year": 2010, "open_date": "2010-10-26", "length_km": 169, "max_speed": 350},
    {"line_name": "福厦铁路", "start_city": "福州", "end_city": "厦门", "open_year": 2010, "open_date": "2010-04-26", "length_km": 275, "max_speed": 250},
    {"line_name": "昌九城际铁路", "start_city": "南昌", "end_city": "九江", "open_year": 2010, "open_date": "2010-09-20", "length_km": 131, "max_speed": 250},
    # 2011年
    {"line_name": "京沪高速铁路", "start_city": "北京", "end_city": "上海", "open_year": 2011, "open_date": "2011-06-30", "length_km": 1318, "max_speed": 380},
    {"line_name": "广深港高铁广深段", "start_city": "广州", "end_city": "深圳", "open_year": 2011, "open_date": "2011-12-26", "length_km": 110, "max_speed": 350},
    # 2012年
    {"line_name": "京广高铁京武段", "start_city": "北京", "end_city": "武汉", "open_year": 2012, "open_date": "2012-12-26", "length_km": 1224, "max_speed": 350},
    {"line_name": "哈大高速铁路", "start_city": "哈尔滨", "end_city": "大连", "open_year": 2012, "open_date": "2012-12-01", "length_km": 921, "max_speed": 350},
    {"line_name": "宁杭高速铁路", "start_city": "南京", "end_city": "杭州", "open_year": 2012, "open_date": "2013-07-01", "length_km": 249, "max_speed": 350},
    # 2013年
    {"line_name": "宁杭高速铁路", "start_city": "南京", "end_city": "杭州", "open_year": 2013, "open_date": "2013-07-01", "length_km": 249, "max_speed": 350},
    {"line_name": "杭甬高速铁路", "start_city": "杭州", "end_city": "宁波", "open_year": 2013, "open_date": "2013-07-01", "length_km": 150, "max_speed": 350},
    {"line_name": "津秦高速铁路", "start_city": "天津", "end_city": "秦皇岛", "open_year": 2013, "open_date": "2013-12-01", "length_km": 261, "max_speed": 350},
    {"line_name": "厦深铁路", "start_city": "厦门", "end_city": "深圳", "open_year": 2013, "open_date": "2013-12-28", "length_km": 514, "max_speed": 250},
    {"line_name": "西宝高速铁路", "start_city": "西安", "end_city": "宝鸡", "open_year": 2013, "open_date": "2013-12-28", "length_km": 148, "max_speed": 250},
    {"line_name": "渝利铁路", "start_city": "重庆", "end_city": "利川", "open_year": 2013, "open_date": "2013-12-28", "length_km": 264, "max_speed": 200},
    # 2014年
    {"line_name": "兰新高速铁路", "start_city": "兰州", "end_city": "乌鲁木齐", "open_year": 2014, "open_date": "2014-12-26", "length_km": 1776, "max_speed": 250},
    {"line_name": "贵广高速铁路", "start_city": "贵阳", "end_city": "广州", "open_year": 2014, "open_date": "2014-12-26", "length_km": 857, "max_speed": 300},
    {"line_name": "南广铁路", "start_city": "南宁", "end_city": "广州", "open_year": 2014, "open_date": "2014-12-26", "length_km": 563, "max_speed": 250},
    {"line_name": "成绵乐城际铁路", "start_city": "成都", "end_city": "乐山", "open_year": 2014, "open_date": "2014-12-20", "length_km": 131, "max_speed": 250},
    {"line_name": "青荣城际铁路", "start_city": "青岛", "end_city": "荣成", "open_year": 2014, "open_date": "2014-12-28", "length_km": 298, "max_speed": 250},
    # 2015年
    {"line_name": "合福高速铁路", "start_city": "合肥", "end_city": "福州", "open_year": 2015, "open_date": "2015-06-28", "length_km": 850, "max_speed": 300},
    {"line_name": "哈齐高速铁路", "start_city": "哈尔滨", "end_city": "齐齐哈尔", "open_year": 2015, "open_date": "2015-08-17", "length_km": 282, "max_speed": 250},
    {"line_name": "成渝高速铁路", "start_city": "成都", "end_city": "重庆", "open_year": 2015, "open_date": "2015-12-26", "length_km": 308, "max_speed": 350},
    {"line_name": "海南环岛高铁", "start_city": "海口", "end_city": "三亚", "open_year": 2015, "open_date": "2015-12-30", "length_km": 653, "max_speed": 250},
    {"line_name": "沈丹高速铁路", "start_city": "沈阳", "end_city": "丹东", "open_year": 2015, "open_date": "2015-09-01", "length_km": 208, "max_speed": 250},
    {"line_name": "吉图珲高速铁路", "start_city": "吉林", "end_city": "珲春", "open_year": 2015, "open_date": "2015-09-20", "length_km": 360, "max_speed": 250},
    # 2016年
    {"line_name": "郑徐高速铁路", "start_city": "郑州", "end_city": "徐州", "open_year": 2016, "open_date": "2016-09-10", "length_km": 362, "max_speed": 350},
    {"line_name": "沪昆高速铁路长昆段", "start_city": "长沙", "end_city": "昆明", "open_year": 2016, "open_date": "2016-12-28", "length_km": 1167, "max_speed": 300},
    {"line_name": "南昆高速铁路", "start_city": "南宁", "end_city": "昆明", "open_year": 2016, "open_date": "2016-12-28", "length_km": 710, "max_speed": 250},
    {"line_name": "京津城际延伸线", "start_city": "天津", "end_city": "滨海新区", "open_year": 2015, "open_date": "2015-09-20", "length_km": 45, "max_speed": 350},
    # 2017年
    {"line_name": "宝兰高速铁路", "start_city": "宝鸡", "end_city": "兰州", "open_year": 2017, "open_date": "2017-07-09", "length_km": 401, "max_speed": 250},
    {"line_name": "西成高速铁路", "start_city": "西安", "end_city": "成都", "open_year": 2017, "open_date": "2017-12-06", "length_km": 658, "max_speed": 250},
    {"line_name": "石济高速铁路", "start_city": "石家庄", "end_city": "济南", "open_year": 2017, "open_date": "2017-12-28", "length_km": 298, "max_speed": 250},
    {"line_name": "武九高速铁路", "start_city": "武汉", "end_city": "九江", "open_year": 2017, "open_date": "2017-09-21", "length_km": 224, "max_speed": 250},
    # 2018年
    {"line_name": "广深港高铁香港段", "start_city": "深圳", "end_city": "香港", "open_year": 2018, "open_date": "2018-09-23", "length_km": 26, "max_speed": 200},
    {"line_name": "济青高速铁路", "start_city": "济南", "end_city": "青岛", "open_year": 2018, "open_date": "2018-12-26", "length_km": 308, "max_speed": 350},
    {"line_name": "杭黄高速铁路", "start_city": "杭州", "end_city": "黄山", "open_year": 2018, "open_date": "2018-12-25", "length_km": 265, "max_speed": 250},
    {"line_name": "怀邵衡铁路", "start_city": "怀化", "end_city": "衡阳", "open_year": 2018, "open_date": "2018-12-26", "length_km": 318, "max_speed": 200},
    {"line_name": "京哈高铁承沈段", "start_city": "承德", "end_city": "沈阳", "open_year": 2018, "open_date": "2018-12-29", "length_km": 506, "max_speed": 350},
    # 2019年
    {"line_name": "商合杭高铁商合段", "start_city": "商丘", "end_city": "合肥", "open_year": 2019, "open_date": "2019-12-01", "length_km": 378, "max_speed": 350},
    {"line_name": "成贵高速铁路", "start_city": "成都", "end_city": "贵阳", "open_year": 2019, "open_date": "2019-12-16", "length_km": 648, "max_speed": 250},
    {"line_name": "京张高速铁路", "start_city": "北京", "end_city": "张家口", "open_year": 2019, "open_date": "2019-12-30", "length_km": 174, "max_speed": 350},
    {"line_name": "昌赣高速铁路", "start_city": "南昌", "end_city": "赣州", "open_year": 2019, "open_date": "2019-12-26", "length_km": 418, "max_speed": 350},
    {"line_name": "日兰高铁日曲段", "start_city": "日照", "end_city": "曲阜", "open_year": 2019, "open_date": "2019-11-26", "length_km": 235, "max_speed": 350},
    {"line_name": "汉十高速铁路", "start_city": "汉口", "end_city": "十堰", "open_year": 2019, "open_date": "2019-11-29", "length_km": 399, "max_speed": 350},
    {"line_name": "郑阜高速铁路", "start_city": "郑州", "end_city": "阜阳", "open_year": 2019, "open_date": "2019-12-01", "length_km": 281, "max_speed": 350},
    # 2020年
    {"line_name": "京雄城际铁路", "start_city": "北京", "end_city": "雄安", "open_year": 2020, "open_date": "2020-12-27", "length_km": 92, "max_speed": 350},
    {"line_name": "银西高速铁路", "start_city": "银川", "end_city": "西安", "open_year": 2020, "open_date": "2020-12-26", "length_km": 618, "max_speed": 250},
    {"line_name": "合安高速铁路", "start_city": "合肥", "end_city": "安庆", "open_year": 2020, "open_date": "2020-12-22", "length_km": 162, "max_speed": 350},
    {"line_name": "郑太高速铁路", "start_city": "郑州", "end_city": "太原", "open_year": 2020, "open_date": "2020-12-12", "length_km": 432, "max_speed": 250},
    # 2021年
    {"line_name": "京哈高铁京沈段", "start_city": "北京", "end_city": "沈阳", "open_year": 2021, "open_date": "2021-01-22", "length_km": 698, "max_speed": 350},
    {"line_name": "徐连高速铁路", "start_city": "徐州", "end_city": "连云港", "open_year": 2021, "open_date": "2021-02-08", "length_km": 180, "max_speed": 350},
    {"line_name": "京港高铁赣深段", "start_city": "赣州", "end_city": "深圳", "open_year": 2021, "open_date": "2021-12-10", "length_km": 436, "max_speed": 350},
    {"line_name": "张吉怀高速铁路", "start_city": "张家界", "end_city": "怀化", "open_year": 2021, "open_date": "2021-12-06", "length_km": 246, "max_speed": 350},
    {"line_name": "沈佳高铁牡佳段", "start_city": "牡丹江", "end_city": "佳木斯", "open_year": 2021, "open_date": "2021-12-06", "length_km": 372, "max_speed": 250},
    {"line_name": "日兰高铁曲庄段", "start_city": "曲阜", "end_city": "庄寨", "open_year": 2021, "open_date": "2021-12-26", "length_km": 204, "max_speed": 350},
    # 2022年
    {"line_name": "济郑高铁济濮段", "start_city": "济南", "end_city": "濮阳", "open_year": 2022, "open_date": "2022-06-20", "length_km": 197, "max_speed": 350},
    {"line_name": "郑渝高铁襄万段", "start_city": "襄阳", "end_city": "万州", "open_year": 2022, "open_date": "2022-06-20", "length_km": 434, "max_speed": 350},
    {"line_name": "京广高铁京武段提速", "start_city": "北京", "end_city": "武汉", "open_year": 2022, "open_date": "2022-06-20", "length_km": 1224, "max_speed": 350},
    {"line_name": "合杭高铁湖杭段", "start_city": "湖州", "end_city": "杭州", "open_year": 2022, "open_date": "2022-09-22", "length_km": 138, "max_speed": 350},
    {"line_name": "弥蒙高速铁路", "start_city": "弥勒", "end_city": "蒙自", "open_year": 2022, "open_date": "2022-12-16", "length_km": 107, "max_speed": 250},
    # 2023年
    {"line_name": "贵南高速铁路", "start_city": "贵阳", "end_city": "南宁", "open_year": 2023, "open_date": "2023-08-31", "length_km": 482, "max_speed": 350},
    {"line_name": "福厦高速铁路", "start_city": "福州", "end_city": "厦门", "open_year": 2023, "open_date": "2023-09-28", "length_km": 277, "max_speed": 350},
    {"line_name": "沪宁沿江高速铁路", "start_city": "南京", "end_city": "太仓", "open_year": 2023, "open_date": "2023-09-28", "length_km": 278, "max_speed": 350},
    {"line_name": "济郑高铁济濮段", "start_city": "济南", "end_city": "郑州", "open_year": 2023, "open_date": "2023-12-08", "length_km": 407, "max_speed": 350},
    {"line_name": "成自宜高速铁路", "start_city": "成都", "end_city": "宜宾", "open_year": 2023, "open_date": "2023-12-26", "length_km": 261, "max_speed": 350},
    {"line_name": "汕汕高铁汕汕段", "start_city": "汕头", "end_city": "汕尾", "open_year": 2023, "open_date": "2023-12-26", "length_km": 162, "max_speed": 350},
    # 2024年
    {"line_name": "池黄高速铁路", "start_city": "池州", "end_city": "黄山", "open_year": 2024, "open_date": "2024-04-26", "length_km": 125, "max_speed": 350},
    {"line_name": "潍烟高速铁路", "start_city": "潍坊", "end_city": "烟台", "open_year": 2024, "open_date": "2024-10-21", "length_km": 237, "max_speed": 350},
    {"line_name": "沪苏湖高速铁路", "start_city": "上海", "end_city": "湖州", "open_year": 2024, "open_date": "2024-12-26", "length_km": 164, "max_speed": 350},
    {"line_name": "南珠高铁南玉段", "start_city": "南宁", "end_city": "玉林", "open_year": 2024, "open_date": "2024-12-30", "length_km": 193, "max_speed": 350},
    # 2025年（部分已规划/在建）
    {"line_name": "西延高速铁路", "start_city": "西安", "end_city": "延安", "open_year": 2025, "open_date": "2025-12-26", "length_km": 299, "max_speed": 350},
    {"line_name": "杭衢高速铁路", "start_city": "杭州", "end_city": "衢州", "open_year": 2025, "open_date": "2025-12-26", "length_km": 131, "max_speed": 350},
]


def crawl_wikipedia_hsr_list() -> List[Dict[str, Any]]:
    """
    爬取 Wikipedia "中国高速铁路线路列表" 页面的数据。
    返回标准化的线路历史记录列表。
    
    注意：Wikipedia 页面结构可能变化，若解析失败请检查页面HTML。
    """
    if not REQUESTS_AVAILABLE:
        print("[爬虫] requests/beautifulsoup4 未安装，跳过 Wikipedia 爬取。")
        return []

    url = "https://zh.wikipedia.org/wiki/中国高速铁路线路列表"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        print(f"[爬虫] 正在请求 Wikipedia: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = "utf-8"
        print(f"[爬虫] 页面获取成功，大小: {len(response.text)} 字符")
    except Exception as e:
        print(f"[爬虫] 请求失败: {e}")
        return []

    soup = BeautifulSoup(response.text, "lxml")
    records = []

    # Wikipedia 页面通常有多个表格，我们需要找到包含线路列表的表格
    # 表格通常包含列：线路名称、长度、设计速度、起点/终点、开通日期等
    tables = soup.find_all("table", {"class": "wikitable"})
    print(f"[爬虫] 找到 {len(tables)} 个 wikitable 表格")

    for table_idx, table in enumerate(tables):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # 解析表头
        headers_row = rows[0]
        header_cells = headers_row.find_all(["th", "td"])
        headers_text = [cell.get_text(strip=True) for cell in header_cells]
        print(f"[爬虫] 表格 {table_idx+1} 表头: {headers_text[:5]}...")

        # 尝试识别包含线路信息的表格
        header_str = "".join(headers_text)
        if not any(keyword in header_str for keyword in ["线路", "名称", "起点", "终点", "开通", "长度"]):
            continue

        print(f"[爬虫] 正在解析表格 {table_idx+1}，共 {len(rows)-1} 行数据...")

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 4:
                continue

            cell_texts = [cell.get_text(strip=True).replace("\n", " ") for cell in cells]

            # 尝试提取信息（根据 Wikipedia 表格结构调整）
            line_name = cell_texts[0] if len(cell_texts) > 0 else ""

            # 查找开通日期（通常包含"年"字的单元格）
            open_date = None
            open_year = None
            for text in cell_texts:
                # 匹配日期格式：YYYY-MM-DD 或 YYYY年MM月DD日
                match_iso = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', text)
                match_cn = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
                if match_iso:
                    open_date = f"{match_iso.group(1)}-{int(match_iso.group(2)):02d}-{int(match_iso.group(3)):02d}"
                    open_year = int(match_iso.group(1))
                    break
                elif match_cn:
                    open_date = f"{match_cn.group(1)}-{int(match_cn.group(2)):02d}-{int(match_cn.group(3)):02d}"
                    open_year = int(match_cn.group(1))
                    break

            # 查找长度（通常包含"公里"或数字）
            length_km = None
            for text in cell_texts:
                match_len = re.search(r'(\d+(?:\.\d+)?)\s*公里', text)
                if match_len:
                    length_km = float(match_len.group(1))
                    break

            # 查找设计速度
            max_speed = None
            for text in cell_texts:
                match_speed = re.search(r'(\d+)\s*km/h', text)
                if match_speed:
                    max_speed = int(match_speed.group(1))
                    break

            # 查找起点/终点（通常包含"—"或"-"）
            start_city = None
            end_city = None
            for text in cell_texts:
                # 匹配 "城市A — 城市B" 或 "城市A-城市B"
                match_route = re.search(r'([^—\-]+)[—\-]([^—\-]+)', text)
                if match_route and not start_city:
                    start_city = match_route.group(1).strip()
                    end_city = match_route.group(2).strip()
                    break

            if line_name and open_year:
                records.append({
                    "line_name": line_name,
                    "start_city": start_city or "",
                    "end_city": end_city or "",
                    "open_year": open_year,
                    "open_date": open_date or f"{open_year}-01-01",
                    "length_km": length_km,
                    "max_speed": max_speed,
                    "source": "wikipedia",
                })

        print(f"[爬虫] 表格 {table_idx+1} 解析完成，提取 {len(records)} 条记录")

    if not records:
        print("[爬虫] 未从 Wikipedia 提取到有效数据，可能页面结构已变化。")
    else:
        print(f"[爬虫] Wikipedia 爬取完成，共 {len(records)} 条记录")

    return records


def merge_with_fallback(crawled_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将爬取的数据与 fallback 数据合并，去重。
    以 line_name + start_city + end_city + open_year 作为唯一键。
    """
    seen = set()
    merged = []

    all_records = crawled_records + FALLBACK_HSR_HISTORY

    for record in all_records:
        key = (record.get("line_name", ""), record.get("start_city", ""), record.get("end_city", ""), record.get("open_year", 0))
        if key not in seen:
            seen.add(key)
            merged.append(record)

    # 按年份排序
    merged.sort(key=lambda x: (x.get("open_year", 9999), x.get("open_date", "")))
    return merged


def save_to_csv(records: List[Dict[str, Any]], filepath: str = "hsr_history.csv"):
    """将数据保存为 CSV 文件。"""
    if not records:
        print("[保存] 无数据可保存")
        return

    fieldnames = ["line_name", "start_city", "end_city", "open_year", "open_date", "length_km", "max_speed", "source"]
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"[保存] 数据已保存至 {filepath}，共 {len(records)} 条记录")


def print_year_summary(records: List[Dict[str, Any]]):
    """按年份统计开通线路数量。"""
    year_counts = {}
    for record in records:
        year = record.get("open_year")
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1

    print("\n[年份统计] 各年开通线路数量：")
    for year in sorted(year_counts.keys()):
        print(f"  {year}年: {year_counts[year]} 条")


def main():
    print("=" * 60)
    print("中国高铁历史线路数据爬虫")
    print("=" * 60)

    # 1. 尝试爬取 Wikipedia
    crawled = crawl_wikipedia_hsr_list()

    # 2. 与 fallback 数据合并
    all_records = merge_with_fallback(crawled)

    # 3. 保存
    save_to_csv(all_records, "hsr_history.csv")

    # 4. 统计
    print_year_summary(all_records)

    print("\n[完成] 数据已准备就绪，可用于动态可视化。")
    print("  下一步：运行主脚本加载 hsr_history.csv 生成时间轴地图。")


if __name__ == "__main__":
    main()
