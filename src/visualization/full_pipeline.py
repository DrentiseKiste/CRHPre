# -*- coding: utf-8 -*-
"""
全量预测完整流程
1. 运行 v2 预测
2. 已有线路 + 预测线路 = 全量多段线路（含途经站）
3. 生成完整地图
"""

import os
import sys
import math
import json
import pandas as pd
from collections import defaultdict

# 添加 src 目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# ============================================
# 第1步：确保有 v2 预测数据
# ============================================
print("="*60)
print("高铁预测系统 - 全量路线 + 途经站")
print("="*60)

data_dir = os.path.join(project_root, "data")
output_dir = os.path.join(project_root, "output")
predicted_file = os.path.join(output_dir, "predicted_routes_v2.csv")
history_file = os.path.join(data_dir, "hsr_history.csv")

if not os.path.exists(predicted_file):
    print("[1/4] 运行预测...")
    import subprocess
    subprocess.run([sys.executable, os.path.join(project_root, "src", "prediction", "hsr_prediction.py")])
else:
    print("[1/4] 预测数据已存在")

# ============================================
# 第2步：计算全量路线（含途经站）
# ============================================
print("\n[2/4] 计算途经站...")
sys.path.insert(0, os.path.join(project_root, "src", "prediction"))
from full_route import FullRoutePredictor

predictor = FullRoutePredictor()

history_df = pd.read_csv(history_file, encoding="utf-8-sig")
existing_routes = predictor.build_full_routes_existing(history_df)
pred_routes = predictor.build_full_routes_predictions()

# 按年组织数据（不累积：每年只保留当年新建/预测的线路）
yearly_data = defaultdict(lambda: {"hist": [], "pred": []})

for year, routes in existing_routes.items():
    yearly_data[year]["hist"] = routes

# 读取预测分数，每年取 Top5（不累加）
v2_df = pd.read_csv(predicted_file)
pred_by_year = defaultdict(list)
for _, row in v2_df.iterrows():
    pred_by_year[int(row['year'])].append({
        'city_a': row['city_a'], 'city_b': row['city_b'],
        'score': row['combined_score']
    })

top5_by_year = {}
for y in sorted(pred_by_year.keys()):
    top5_by_year[y] = sorted(pred_by_year[y], key=lambda x: x['score'], reverse=True)[:5]
    names = [f"{p['city_a']}-{p['city_b']}" for p in top5_by_year[y]]
    print(f"  {y}年 Top5: {', '.join(names)}")

# 每年只保留当年预测 Top5（不累加）
all_years = list(range(2008, 2031))
for y in sorted(pred_routes.keys()):
    top_pairs = {(p['city_a'], p['city_b']) for p in top5_by_year.get(y, [])}
    top_pairs.update({(p['city_b'], p['city_a']) for p in top5_by_year.get(y, [])})
    this_year = []
    for route in pred_routes.get(y, []):
        if len(route) >= 2 and (route[0], route[-1]) in top_pairs:
            this_year.append(route)
    yearly_data[y]["pred"] = this_year

total_hist = sum(len(yearly_data[y]["hist"]) for y in all_years)
total_pred = sum(len(yearly_data[y]["pred"]) for y in all_years)
print(f"  已有线路: {total_hist}条 (含途经站 {sum(1 for y in all_years for r in yearly_data[y]['hist'] if len(r)>2)})")
print(f"  预测线路: {total_pred}条 (含途经站 {sum(1 for y in all_years for r in yearly_data[y]['pred'] if len(r)>2)})")

# ============================================
# 第3步：生成地图（LayerControl 勾选框，每年独立）
# ============================================
print("\n[3/4] 生成地图...")

sys.path.insert(0, os.path.join(project_root, "src", "prediction"))
from china_hsr_prediction import CITIES_DATA
cities_dict = {c['name']: c for c in CITIES_DATA}
city_map = {c['name']: (c['lat'], c['lon']) for c in CITIES_DATA}

# 城市标记 JS
city_markers_js = ""
max_pop = max(c['population'] for c in CITIES_DATA)
for c in CITIES_DATA:
    r = 3 + (c['population'] / max_pop) * 12
    admin_colors = {
        "直辖市": "#d62728", "特别行政区": "#9467bd",
        "省会/副省级": "#ff7f0e", "地级市": "#2ca02c", "县级": "#1f77b4"}
    color = admin_colors.get(c.get('admin_level', ''), "#7f7f7f")
    city_markers_js += f'''
    L.circleMarker([{c['lat']},{c['lon']}], {{
        radius: {r:.1f}, color: "{color}", fillColor: "{color}",
        fillOpacity: 0.7
    }}).bindPopup("<b>{c['name']}</b><br>GDP: {c['gdp']}亿")
      .addTo(cityGroup);'''

def route_to_coords(route):
    coords = []
    for s in route:
        if s in city_map:
            coords.append([city_map[s][0], city_map[s][1]])
    return coords if len(coords) >= 2 else None

# 生成历史图层 JS（每年独立）
hist_layer_init_js = ""
hist_layer_fill_js = ""
for y in all_years:
    routes = yearly_data[y]["hist"]
    if not routes:
        continue
    hcnt = len(routes)
    hist_layer_init_js += f'''
var histGroup{y} = L.layerGroup();
overlayLayers['{y}年 历史 ({hcnt}条)'] = histGroup{y};'''
    hdata = []
    for route in routes:
        coords = route_to_coords(route)
        if coords:
            hdata.append({
                "coords": coords,
                "popup": " → ".join([s for s in route if s in city_map])
            })
    hj = json.dumps(hdata, ensure_ascii=False)
    hist_layer_fill_js += f'''
(function() {{
  var d = {hj};
  d.forEach(function(r) {{
    L.polyline(r.coords, {{color: '#666666', weight: 1.8, opacity: 0.5}})
      .bindPopup(r.popup).addTo(histGroup{y});
  }});
}})();'''

# 生成预测图层 JS（每年独立 Top5）
pred_layer_init_js = ""
pred_layer_fill_js = ""
for y in all_years:
    routes = yearly_data[y]["pred"]
    if not routes:
        continue
    pcnt = len(routes)
    pred_layer_init_js += f'''
var predGroup{y} = L.layerGroup();
overlayLayers['{y}年 预测 ({pcnt}条)'] = predGroup{y};'''
    pdata = []
    for route in routes:
        coords = route_to_coords(route)
        if coords:
            pdata.append({
                "coords": coords,
                "popup": "预测: " + " → ".join([s for s in route if s in city_map])
            })
    pj = json.dumps(pdata, ensure_ascii=False)
    pred_layer_fill_js += f'''
(function() {{
  var d = {pj};
  d.forEach(function(r) {{
    L.polyline(r.coords, {{color: '#e41a1c', weight: 2.5, opacity: 0.8, dashArray: '6 4'}})
      .bindPopup(r.popup).addTo(predGroup{y});
  }});
}})();'''

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>中国高铁全量路线图 (2008-2030)</title>
<link rel="stylesheet" href="https://cdn.bootcdn.net/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdn.bootcdn.net/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>if(typeof L==='undefined'){{document.write('<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.js"><\\/script>');}}</script>
<style>
  body {{ margin:0; padding:0; font-family: "Microsoft YaHei", sans-serif; }}
  #map {{ width:100vw; height:100vh; background:#e8e8e8; }}
  .info-panel {{
    position:fixed; top:10px; left:50px; z-index:9999;
    background:white; padding:15px; border:2px solid #333;
    border-radius:5px; font-size:14px; max-width:300px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }}
  .info-panel h3 {{ margin:0 0 8px 0; color:#2c3e50; font-size:16px; }}
  .legend {{ color:#555; font-size:12px; margin-top:6px; line-height:1.8; }}
  .legend-line {{ display:flex; align-items:center; gap:8px; }}
  .legend-swatch {{ width:16px; height:16px; border-radius:3px; display:inline-block; flex-shrink:0; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="info-panel">
  <h3>🚄 中国高铁全量路线图 (2008-2030)</h3>
  <div class="legend">
    <div class="legend-line">
      <span class="legend-swatch" style="background:#666"></span> 灰色=当年建成线路
    </div>
    <div class="legend-line">
      <span class="legend-swatch" style="background:#e41a1c"></span> 红色虚线=当年预测 (Top5)
    </div>
    <div style="font-size:11px;color:#999;margin-top:4px;">
      💡 右侧勾选年份查看 · 默认仅显示城市
    </div>
  </div>
</div>

<script>
if (typeof L === 'undefined') {{
  document.getElementById('map').innerHTML =
    '<div style="text-align:center;padding-top:200px;font-size:18px;color:#c00;">'+
    '⚠️ 地图库加载失败<br><span style="font-size:14px;color:#666;">请用浏览器直接打开此HTML文件</span></div>';
}} else {{

var map = L.map('map', {{zoomControl: true}}).setView([35, 105], 5);

var tileLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '&copy; <a href="https://osm.org">OSM</a>',
  maxZoom: 12,
  subdomains: 'abc'
}}).addTo(map);

tileLayer.on('tileerror', function() {{
  if (!tileLayer._fallbackTried) {{
    tileLayer._fallbackTried = true;
    tileLayer.setUrl('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png');
  }}
}});

var cityGroup = L.layerGroup().addTo(map);
{city_markers_js}

var overlayLayers = {{}};

{hist_layer_init_js}
{hist_layer_fill_js}

{pred_layer_init_js}
{pred_layer_fill_js}

L.control.layers(null, overlayLayers, {{position: 'topright', collapsed: false}}).addTo(map);
}}
</script>
</body>
</html>'''

output_path = os.path.join(output_dir, "china_hsr_full_routes.html")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"  地图已保存: {output_path}")

# ============================================
# 第4步：统计
# ============================================
print("\n[4/4] 统计报告")
print("-"*40)
for y in all_years:
    h = yearly_data[y]["hist"]
    p = yearly_data[y]["pred"]
    h_with_via = sum(1 for r in h if len(r) > 2)
    p_with_via = sum(1 for r in p if len(r) > 2)
    if h or p:
        print(f"  {y}年: 历史{h_with_via}/{len(h)}途经 + 预测{p_with_via}/{len(p)}途经")

print(f"\n{'='*60}")
print("完成！请打开 china_hsr_full_routes.html")
print("="*60)