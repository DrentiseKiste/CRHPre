
# -*- coding: utf-8 -*-
"""
中国高铁动态时间轴可视化（2008-2030）
【功能分离】：预测功能已分离到 hsr_rolling_prediction.py
本程序从 predicted_routes_rolling.csv 读取预测结果
"""
import pandas as pd
import folium
from folium import CircleMarker, PolyLine, FeatureGroup

from china_hsr_prediction import CITIES_DATA
from china_hsr_prediction import build_city_df


def load_hsr_history(csv_path="hsr_history.csv"):
    """加载历史高铁线路数据"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df['open_year'] = pd.to_numeric(df['open_year'], errors='coerce').fillna(0).astype(int)
        print("[历史数据] 加载了", len(df), "条线路记录")
        return df
    except FileNotFoundError:
        print("[警告] 未找到", csv_path)
        return pd.DataFrame()


def load_predicted_routes(csv_path="predicted_hsr_routes_2026_2050.csv"):
    """从预测程序生成的文件中加载预测线路"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print("[预测数据] 加载了", len(df), "条预测线路")
        return df
    except FileNotFoundError:
        print("[警告] 未找到", csv_path, "，请先运行 advanced_hsr_prediction.py 生成预测结果")
        return pd.DataFrame()


def build_year_routes(history_df):
    """按年份整理线路"""
    city_names = set()
    for city in CITIES_DATA:
        city_names.add(city['name'])
    
    year_routes = {}

    for idx, row in history_df.iterrows():
        year = int(row['open_year'])
        start = str(row['start_city']).strip()
        end = str(row['end_city']).strip()

        if start not in city_names or end not in city_names:
            continue

        if year not in year_routes:
            year_routes[year] = []
        year_routes[year].append((start, end))

    return year_routes


def build_yearly_incremental_prediction(year_routes, predict_df=None, top_n=10):
    """
    构建逐年递增的预测线路
    返回: {年份: (历史线路列表, 该年新增预测线路列表)}

    策略：
    - 2026年：新增Top 10预测线路中的前3条
    - 2027年：新增接下来的2条
    - 2028年：新增接下来的2条
    - 2029年：新增剩下的3条
    - 2030年：新增0条
    """
    cumulative = {}

    # 历史累积
    accumulated_hist = set()
    for year in sorted(year_routes.keys()):
        for route in year_routes[year]:
            normalized = tuple(sorted(route))
            accumulated_hist.add(normalized)
        cumulative[year] = (list(accumulated_hist), [])

    # 预测线路逐年递增
    if predict_df is not None and len(predict_df) > 0:
        last_year = max(year_routes.keys()) if year_routes else 2025

        # 从预测文件获取线路（按概率排序）
        predict_list = []
        for idx, row in predict_df.head(top_n).iterrows():
            normalized = tuple(sorted([row['city_a'], row['city_b']]))
            predict_list.append(normalized)

        # 定义逐年增加策略
        yearly_addition = {
            2026: 3,   # 2026年新增3条
            2027: 2,   # 2027年新增2条
            2028: 2,   # 2028年新增2条
            2029: 3,   # 2029年新增3条
        }

        # 逐年预测线路（只记录每年新增的）
        predict_index = 0

        for future_year in range(last_year + 1, 2031):
            # 计算该年应新增的线路数
            add_count = yearly_addition.get(future_year, 0)

            # 该年新增的预测线路
            new_pred_routes = []
            for _ in range(add_count):
                if predict_index < len(predict_list):
                    new_pred_routes.append(predict_list[predict_index])
                    predict_index += 1

            # 只返回当年新增的预测线路，不累积
            cumulative[future_year] = (list(accumulated_hist), new_pred_routes)

    return cumulative


def create_map(city_df, cumulative_data, output_path="china_hsr_timeline_simple.html"):
    """创建地图"""
    m = folium.Map(location=[35.0, 105.0], zoom_start=5, tiles="CartoDB positron")
    city_map = {}
    for idx, row in city_df.iterrows():
        city_map[row['name']] = (row['lat'], row['lon'])

    # 添加城市圆
    max_pop = city_df['population'].max()
    admin_colors = {
        "直辖市": "#d62728",
        "特别行政区": "#9467bd",
        "省会/副省级": "#ff7f0e",
        "地级市": "#2ca02c",
        "县级": "#1f77b4",
    }

    for idx, row in city_df.iterrows():
        radius = 3 + (row['population'] / max_pop) * 12
        color = admin_colors.get(row['admin_level'], "#7f7f7f")

        CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>{row['name']}</b><br>GDP: {row['gdp']}亿<br>人口: {row['population']}万"
        ).add_to(m)

    # 创建年份图层组
    all_years = sorted(cumulative_data.keys())

    for year in all_years:
        hist_routes, pred_routes = cumulative_data[year]

        # 历史线路图层
        if hist_routes:
            hist_group = FeatureGroup(name=f"{year}年 历史({len(hist_routes)}条)", show=False)
            for route in hist_routes:
                start_city, end_city = route
                if start_city not in city_map or end_city not in city_map:
                    continue
                coords = [city_map[start_city], city_map[end_city]]
                PolyLine(
                    locations=coords,
                    color="#555555",
                    weight=1.5,
                    opacity=0.4,
                    popup=f"⚫ 历史: {start_city} ↔ {end_city}"
                ).add_to(hist_group)
            hist_group.add_to(m)

        # 预测线路图层
        if pred_routes:
            pred_group = FeatureGroup(name=f"{year}年 预测({len(pred_routes)}条)", show=False)
            for route in pred_routes:
                start_city, end_city = route
                if start_city not in city_map or end_city not in city_map:
                    continue
                coords = [city_map[start_city], city_map[end_city]]
                PolyLine(
                    locations=coords,
                    color="#e41a1c",
                    weight=2.5,
                    opacity=0.8,
                    popup=f"🔴 预测: {start_city} ↔ {end_city}"
                ).add_to(pred_group)
            pred_group.add_to(m)

    # 添加图层控制
    folium.LayerControl(collapsed=False).add_to(m)

    # 统计信息
    stats = []
    for year in all_years:
        hist_len = len(cumulative_data[year][0])
        pred_len = len(cumulative_data[year][1])
        stats.append((year, hist_len, pred_len))

    # 年份选择器
    year_options = ""
    for y, h, p in stats:
        year_options += f'<option value="{y}">{y}年 (历史:{h}条, 预测:{p}条)</option>'

    selector_html = f'''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999; background-color: white;
                padding: 15px; border: 2px solid grey; border-radius: 5px; font-size: 14px; max-width: 380px;">
        <h4 style="margin: 0 0 10px 0;">🚄 中国高铁动态网络 (2008-2030)</h4>
        <p style="margin: 0 0 10px 0; font-size: 12px; color: #666;">
            选择年份查看该年高铁网络状态<br>
            <span style="color: #555555;">■</span> 灰色=历史线路 |
            <span style="color: #e41a1c;">■</span> 红色=预测线路
        </p>
        <select id="yearSelector" onchange="toggleYear(this.value)" style="padding: 8px; font-size: 14px; width: 100%;">
            {year_options}
        </select>
        <div style="margin-top: 10px; font-size: 11px; color: #888;">
            💡 预测线路逐年递增：2026年3条→2027年5条→2028年7条→2029年10条
        </div>
    </div>
    <script>
    var years = {all_years};

    function toggleYear(selectedYear) {{
        var yearInt = parseInt(selectedYear);

        document.querySelectorAll('.leaflet-control-layers label').forEach(function(label) {{
            label.style.display = 'none';
            var checkbox = label.querySelector('input');
            if (checkbox) checkbox.checked = false;
        }});

        var layers = document.querySelectorAll('.leaflet-control-layers label');
        layers.forEach(function(label) {{
            var name = label.textContent || label.innerText;
            if (name.startsWith(yearInt + '年 历史') || name.startsWith(yearInt + '年 预测')) {{
                label.style.display = 'block';
                var checkbox = label.querySelector('input');
                if (checkbox) checkbox.checked = true;
            }}
        }});
    }}

    toggleYear(2008);
    </script>
    '''

    m.get_root().html.add_child(folium.Element(selector_html))

    m.save(output_path)
    print("[地图已保存]", output_path)

    print("\n每年线路统计：")
    for year, hist, pred in stats:
        if year in [2008, 2010, 2015, 2020, 2025, 2026, 2027, 2028, 2029, 2030]:
            print(f"  {year}年: {hist}条历史 + {pred}条预测")

    return m


def main():
    print("=" * 60)
    print("中国高铁动态时间轴可视化（2008-2030）")
    print("=" * 60)
    print("\n【功能说明】")
    print("本程序已分离预测功能，预测由 hsr_rolling_prediction.py 处理")
    print("本程序从 predicted_routes_rolling.csv 读取预测结果")
    print("\n【预测线路递增策略】")
    print("2026年: 新增Top10中的前3条")
    print("2027年: 新增前5条")
    print("2028年: 新增前7条")
    print("2029年+: 全部10条")

    city_df = build_city_df(CITIES_DATA)
    print(f"\n[数据加载] {len(city_df)} 个城市")

    history_df = load_hsr_history("hsr_history.csv")
    year_routes = build_year_routes(history_df)
    if year_routes:
        print(f"[历史线路] 覆盖 {min(year_routes.keys())}-{max(year_routes.keys())}")

    predict_df = load_predicted_routes("predicted_routes_rolling.csv")

    # 使用逐年递增的预测线路
    cumulative_data = build_yearly_incremental_prediction(year_routes, predict_df, top_n=10)

    create_map(city_df, cumulative_data, "china_hsr_timeline_simple.html")

    print("\n[完成] 请打开 china_hsr_timeline_simple.html")


if __name__ == "__main__":
    main()

