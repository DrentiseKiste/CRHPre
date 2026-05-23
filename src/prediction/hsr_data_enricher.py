# -*- coding: utf-8 -*-
"""
高铁线路数据补充器
为现有线路补充途经的中间高铁站

例如：京沪高铁 北京→上海 补充为 北京→天津→济南→南京→上海
这样预测时就能知道天津-济南已经被京沪高铁覆盖，无需重复预测

注意：只补充城市列表中已有的高铁站，不新增站点
"""

import pandas as pd
import json
import os
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_dir = os.path.join(project_root, "data")


class HSRDataEnricher:
    """数据补充器：为线路添加中间站"""

    def __init__(self):
        # 加载城市列表，确保只使用已有的站点
        self.valid_stations = set()
        if True:
            from china_hsr_prediction import CITIES_DATA
            for c in CITIES_DATA:
                self.valid_stations.add(c['name'])

        # 预定义各主要干线的途经站点（八纵八横 + 已建成线路）
        # 只包含 valid_stations 中已有的站点
        self.line_stations = self._build_line_stations()

    def _build_line_stations(self):
        """构建线路-途经站点的完整映射"""
        raw = {
            # ===== 八纵 =====
            "京哈高铁": ["北京", "承德", "沈阳", "四平", "长春", "哈尔滨"],
            "京沪高铁": ["北京", "天津", "济南", "曲阜", "徐州", "南京", "苏州", "上海"],
            "京港高铁": ["北京", "雄安", "石家庄", "郑州", "武汉", "长沙",
                       "衡阳", "赣州", "深圳", "香港"],
            "京昆通道": ["北京", "雄安", "石家庄", "太原", "西安", "成都", "昆明"],
            "沿海通道": ["大连", "丹东", "沈阳", "秦皇岛", "天津", "烟台",
                       "青岛", "连云港", "上海", "杭州", "宁波", "温州",
                       "福州", "厦门", "汕头", "深圳"],
            "陆桥通道": ["连云港", "徐州", "郑州", "西安", "宝鸡", "兰州", "乌鲁木齐"],
            "沿江通道": ["上海", "苏州", "南京", "合肥", "武汉", "宜昌", "重庆", "成都"],
            "沪昆通道": ["上海", "杭州", "南昌", "长沙", "贵阳", "昆明"],

            # ===== 八横 =====
            "绥满通道": ["哈尔滨", "齐齐哈尔"],
            "京兰通道": ["北京", "张家口", "呼和浩特", "包头", "银川", "兰州"],
            "青银通道": ["青岛", "潍坊", "济南", "石家庄", "太原", "银川"],
            "厦渝通道": ["厦门", "赣州", "长沙", "张家界", "重庆"],

            # ===== 重要区域线路 =====
            "兰广通道": ["兰州", "成都", "贵阳", "广州"],
            "广昆通道": ["广州", "南宁", "昆明"],

            # ===== 已建成重要线路 =====
            "武广高铁": ["武汉", "长沙", "衡阳", "广州"],
            "郑西高铁": ["郑州", "洛阳", "西安"],
            "哈大高铁": ["哈尔滨", "长春", "沈阳", "大连"],
            "西成高铁": ["西安", "汉中", "成都"],
            "成渝高铁": ["成都", "内江", "重庆"],
            "贵广高铁": ["贵阳", "桂林", "广州"],
            "南广铁路": ["南宁", "梧州", "广州"],
            "合福高铁": ["合肥", "黄山", "福州"],
            "沪宁城际": ["上海", "苏州", "无锡", "南京"],
            "杭甬高铁": ["杭州", "绍兴", "宁波"],
            "津秦高铁": ["天津", "唐山", "秦皇岛"],
            "厦深铁路": ["厦门", "汕头", "深圳"],
            "郑徐高铁": ["郑州", "商丘", "徐州"],
            "宝兰高铁": ["宝鸡", "天水", "兰州"],
            "武九高铁": ["武汉", "九江"],
            "石济高铁": ["石家庄", "衡水", "济南"],
            "济青高铁": ["济南", "淄博", "潍坊", "青岛"],
            "杭黄高铁": ["杭州", "黄山"],
            "成贵高铁": ["成都", "宜宾", "贵阳"],
            "昌赣高铁": ["南昌", "吉安", "赣州"],
            "京张高铁": ["北京", "张家口"],
            "银西高铁": ["银川", "西安"],
            "京雄城际": ["北京", "雄安"],
            "贵南高铁": ["贵阳", "南宁"],
            "张吉怀": ["张家界", "吉首", "怀化"],
            "郑阜高铁": ["郑州", "周口", "阜阳"],
            "商合杭": ["商丘", "阜阳", "合肥", "芜湖", "杭州"],
            "弥蒙高铁": ["弥勒", "蒙自"],
            "济郑高铁": ["济南", "郑州"],
            "福厦高铁": ["福州", "厦门"],
            "池黄高铁": ["池州", "黄山"],
            "潍烟高铁": ["潍坊", "烟台"],
            "沪苏湖": ["上海", "苏州", "湖州"],
        }

        # 过滤：只保留 valid_stations 中存在的站点
        filtered = {}
        for line_name, stations in raw.items():
            valid_stops = [s for s in stations if s in self.valid_stations]
            if len(valid_stops) >= 2:
                filtered[line_name] = valid_stops

        return filtered

    def generate_enriched_csv(self, output_path=None):
        """生成补充后的线路数据"""
        if output_path is None:
            output_path = os.path.join(data_dir, "hsr_history_enriched.csv")
        df = pd.read_csv(os.path.join(data_dir, "hsr_history.csv"), encoding="utf-8-sig")
        df['via_stations'] = ""

        for idx, row in df.iterrows():
            line_name = row['line_name']
            start = row['start_city']
            end = row['end_city']

            # 查找匹配的线路途经站
            via = self._find_via_stations(line_name, start, end)
            df.at[idx, 'via_stations'] = via

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✓ 补充后数据已保存: {output_path}")
        return df

    def _find_via_stations(self, line_name, start_city, end_city):
        """查找线路的途经站点"""
        # 精确匹配线路名
        if line_name in self.line_stations:
            stations = self.line_stations[line_name]
            if start_city in stations and end_city in stations:
                idx_start = stations.index(start_city)
                idx_end = stations.index(end_city)
                if abs(idx_start - idx_end) > 1:
                    via = stations[min(idx_start, idx_end)+1:max(idx_start, idx_end)]
                    return "→".join(via)

        # 模糊匹配：查找包含首尾站的线路
        for lname, stations in self.line_stations.items():
            if start_city in stations and end_city in stations:
                idx_start = stations.index(start_city)
                idx_end = stations.index(end_city)
                if abs(idx_start - idx_end) > 1:
                    via = stations[min(idx_start, idx_end)+1:max(idx_start, idx_end)]
                    return "→".join(via)

        return ""

    def build_topology_graph(self):
        """
        构建包含中间站的完整拓扑图
        用于预测时的冲突检测
        """
        from simple_topology import SimpleHSRTopology

        topology = SimpleHSRTopology()

        # 添加所有预定义线路
        for line_name, stations in self.line_stations.items():
            topology.add_line(line_name, stations)

        return topology

    def get_existing_coverage(self):
        """
        获取所有已有线路覆盖的城市对
        包括直接相连和通过中间站相连的

        例如：京沪高铁 北京→上海
        覆盖：北京-天津, 天津-济南, 济南-南京, 南京-上海（直接段）
        也覆盖：北京-济南, 天津-南京, 北京-上海 等（间接覆盖）
        """
        covered_pairs = defaultdict(set)  # (city_a, city_b) -> {line_name}

        for line_name, stations in self.line_stations.items():
            for i in range(len(stations)):
                for j in range(i + 1, len(stations)):
                    pair = tuple(sorted([stations[i], stations[j]]))
                    covered_pairs[pair].add(line_name)

        return covered_pairs

    def print_summary(self):
        """打印补充统计"""
        df = pd.read_csv("hsr_history.csv", encoding="utf-8-sig")
        enriched = self.generate_enriched_csv("hsr_history_enriched.csv")

        enriched_count = sum(1 for v in enriched['via_stations'] if v)
        print(f"\n线路总数: {len(enriched)}")
        print(f"有途经站的线路: {enriched_count}")
        print(f"无途经站的线路: {len(enriched) - enriched_count}")

        print(f"\n有途经站的线路示例:")
        for _, row in enriched.iterrows():
            if row['via_stations']:
                print(f"  {row['line_name']}: {row['start_city']}→{row['via_stations']}→{row['end_city']}")

        # 覆盖统计
        covered = self.get_existing_coverage()
        print(f"\n已有线路覆盖的城市对总数: {len(covered)}")
        return enriched


if __name__ == "__main__":
    enricher = HSRDataEnricher()
    enriched_df = enricher.print_summary()