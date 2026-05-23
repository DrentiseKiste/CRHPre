# -*- coding: utf-8 -*-
"""
全量路线预测 — 已有线路和预测线路都标出途经站
1. 已有线路：从数据补充器获取途经站，画多段线
2. 预测线路：自动计算途经站（主干直线优先，偏移<15%）
"""

import math
import os
import pandas as pd
from collections import defaultdict

from hsr_data_enricher import HSRDataEnricher
from china_hsr_prediction import CITIES_DATA

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_dir = os.path.join(project_root, "output")


class FullRoutePredictor:
    """全量路线预测器"""

    def __init__(self):
        self.enricher = HSRDataEnricher()
        self.cities = {c['name']: c for c in CITIES_DATA}

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def point_to_line_distance(self, px, py, ax, ay, bx, by):
        """点(px,py)到线段(ax,ay)-(bx,by)的垂直距离(km)"""
        # 转换到笛卡尔近似（对于中国范围内的短距离足够精确）
        lat_mid = (ay + by) / 2
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * math.cos(math.radians(lat_mid))

        px2 = px * km_per_deg_lat
        py2 = py * km_per_deg_lon
        ax2 = ax * km_per_deg_lat
        ay2 = ay * km_per_deg_lon
        bx2 = bx * km_per_deg_lat
        by2 = by * km_per_deg_lon

        # 线段向量
        ABx = bx2 - ax2
        ABy = by2 - ay2
        APx = px2 - ax2
        APy = py2 - ay2

        len_sq = ABx*ABx + ABy*ABy
        if len_sq == 0:
            return math.sqrt(APx*APx + APy*APy)

        t = max(0, min(1, (APx*ABx + APy*ABy) / len_sq))
        proj_x = ax2 + t * ABx
        proj_y = ay2 + t * ABy

        return math.sqrt((px2-proj_x)**2 + (py2-proj_y)**2)

    def find_intermediate_stations(self, city_a, city_b, max_deviation=0.15):
        """
        为预测线路(A→B)寻找合理的中间站
        条件：
        1. 在A-B直线上或附近（偏移 < max_deviation * 总距离）
        2. 位于A和B之间（参数t在0到1之间）
        3. 优先选择行政级别高的城市
        """
        if city_a not in self.cities or city_b not in self.cities:
            return []

        a = self.cities[city_a]
        b = self.cities[city_b]
        total_dist = self.haversine(a['lat'], a['lon'], b['lat'], b['lon'])

        max_dev_km = total_dist * max_deviation

        candidates = []
        for name, c in self.cities.items():
            if name in (city_a, city_b):
                continue

            dev = self.point_to_line_distance(
                c['lat'], c['lon'],
                a['lat'], a['lon'],
                b['lat'], b['lon'])

            if dev > max_dev_km:
                continue

            # 计算t参数（0~1之间表示在A和B之间）
            lat_mid = (a['lat'] + b['lat']) / 2
            km_per_deg_lat = 111.32
            km_per_deg_lon = 111.32 * math.cos(math.radians(lat_mid))

            ax2, ay2 = a['lat']*km_per_deg_lat, a['lon']*km_per_deg_lon
            bx2, by2 = b['lat']*km_per_deg_lat, b['lon']*km_per_deg_lon
            cx2, cy2 = c['lat']*km_per_deg_lat, c['lon']*km_per_deg_lon

            ABx = bx2 - ax2
            ABy = by2 - ay2
            ACx = cx2 - ax2
            ACy = cy2 - ay2

            len_sq = ABx*ABx + ABy*ABy
            if len_sq == 0:
                continue
            t = (ACx*ABx + ACy*ABy) / len_sq

            if t <= 0.05 or t >= 0.95:
                continue

            # 评分：行政级别 + 靠近中线
            admin_vals = {'直辖市': 5, '特别行政区': 5,
                          '省会/副省级': 4, '地级市': 3, '县级': 2}
            admin_score = admin_vals.get(c.get('admin_level', ''), 2) / 5.0
            line_score = 1.0 - dev / max_dev_km

            score = 0.4 * admin_score + 0.6 * line_score
            candidates.append((name, t, score, dev))

        # 按t排序（沿线路方向）
        candidates.sort(key=lambda x: x[1])

        # 过滤太近的站（间距<50km可合并）
        filtered = []
        for item in candidates:
            if not filtered:
                filtered.append(item)
            else:
                prev_t = filtered[-1][1]
                if item[1] - prev_t > 0.03:  # 至少间隔3%的总距离
                    filtered.append(item)

        # 限制中间站数量（3-6个）
        if len(filtered) <= 6:
            return filtered
        else:
            # 按分数排序取top6，再按位置排序
            top = sorted(filtered, key=lambda x: x[2], reverse=True)[:6]
            return sorted(top, key=lambda x: x[1])

    def build_full_routes_existing(self, history_df):
        """构建已有线路的完整路由（含途经站）"""
        full_routes = {}  # year -> [full_route_list]

        year_groups = defaultdict(list)
        for _, row in history_df.iterrows():
            year = int(row['open_year'])
            line_name = row['line_name']
            start = row['start_city']
            end = row['end_city']

            # 从数据补充器获取途经站
            via_str = self.enricher._find_via_stations(line_name, start, end)
            if via_str:
                stations = [start] + via_str.split('→') + [end]
            else:
                stations = [start, end]

            year_groups[year].append(stations)

        return dict(year_groups)

    def build_full_routes_predictions(self):
        """为预测线路计算完整路由（含中间站）"""
        v2 = pd.read_csv(os.path.join(output_dir, "predicted_routes_v2.csv"))

        full_routes = {}  # year -> [full_route_list]

        for _, row in v2.iterrows():
            year = int(row['year'])
            city_a = row['city_a']
            city_b = row['city_b']

            intermediates = self.find_intermediate_stations(city_a, city_b)

            route = [city_a]
            for name, t, score, dev in intermediates:
                route.append(name)
            route.append(city_b)

            if year not in full_routes:
                full_routes[year] = []
            full_routes[year].append(route)

        return full_routes

    def run(self):
        """运行全量路线预测"""
        print("="*60)
        print("全量路线预测")
        print("="*60)

        # 已有线路
        history_df = pd.read_csv("hsr_history.csv", encoding="utf-8-sig")
        existing_routes = self.build_full_routes_existing(history_df)

        total_existing = sum(len(v) for v in existing_routes.values())
        with_via = sum(1 for routes in existing_routes.values()
                       for r in routes if len(r) > 2)
        print(f"已有线路: {total_existing}条, {with_via}条含途经站")

        # 预测线路
        pred_routes = self.build_full_routes_predictions()
        total_pred = sum(len(v) for v in pred_routes.values())
        with_via_pred = sum(1 for routes in pred_routes.values()
                            for r in routes if len(r) > 2)
        print(f"预测线路: {total_pred}条, {with_via_pred}条含途经站")

        # 示例
        print("\n已有线路示例（多段线）:")
        for year in sorted(existing_routes.keys())[:3]:
            for route in existing_routes[year][:2]:
                if len(route) > 2:
                    print(f"  {year}年: {' → '.join(route)}")

        print("\n预测线路示例（含途经站）:")
        for year in sorted(pred_routes.keys()):
            for route in pred_routes[year][:3]:
                parts = [f"{route[i]}→{route[i+1]}"
                         for i in range(len(route)-1)]
                print(f"  {year}年: {' → '.join(route)}")
                for seg in parts:
                    # 计算段距离
                    if route[i] in self.cities and route[i+1] in self.cities:
                        a = self.cities[route[i]]
                        b = self.cities[route[i+1]]
                        d = self.haversine(a['lat'], a['lon'], b['lat'], b['lon'])
                        print(f"         段 {seg}: {d:.0f}km")
            break  # 只展示一年

        return existing_routes, pred_routes


if __name__ == "__main__":
    pred = FullRoutePredictor()
    existing, predicted = pred.run()