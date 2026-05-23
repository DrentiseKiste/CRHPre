
# -*- coding: utf-8 -*-
"""
简化版高铁线路拓扑结构模块
"""

from collections import defaultdict
import networkx as nx


class SimpleHSRTopology:
    """简化版高铁拓扑结构"""
    
    def __init__(self):
        self.lines = {}
        self.station_to_lines = defaultdict(set)
        self.graph = nx.Graph()
    
    def add_line(self, line_name, stations):
        """添加一条线路"""
        self.lines[line_name] = stations
        
        for station in stations:
            self.station_to_lines[station].add(line_name)
        
        for i in range(len(stations)-1):
            a = stations[i]
            b = stations[i+1]
            self.graph.add_edge(a, b, line=line_name)
    
    def is_direct_segment(self, city_a, city_b):
        """检查是否是直接线段"""
        return self.graph.has_edge(city_a, city_b)
    
    def find_connecting_lines(self, city_a, city_b):
        """查找连接两个城市的线路"""
        common_lines = set()
        if city_a in self.station_to_lines and city_b in self.station_to_lines:
            common_lines = self.station_to_lines[city_a] &amp; self.station_to_lines[city_b]
        
        valid_lines = []
        for line_name in common_lines:
            stations = self.lines[line_name]
            if city_a in stations and city_b in stations:
                valid_lines.append(line_name)
        
        return valid_lines
    
    def check_conflicts(self, city_a, city_b):
        """检查冲突"""
        conflicts = []
        
        for line_name, stations in self.lines.items():
            if city_a in stations and city_b in stations:
                idx_a = stations.index(city_a)
                idx_b = stations.index(city_b)
                if abs(idx_a - idx_b) == 1:
                    conflicts.append({
                        "type": "redundant_segment",
                        "line": line_name,
                        "description": "线路 " + line_name + " 已包含直接线段 " + city_a + "-" + city_b,
                        "severity": 5
                    })
                else:
                    conflicts.append({
                        "type": "existing_path",
                        "line": line_name,
                        "description": "线路 " + line_name + " 已覆盖 " + city_a + " 到 " + city_b + " 的完整路径",
                        "severity": 3
                    })
        
        return conflicts
    
    def is_pair_connected(self, city_a, city_b):
        """检查两个城市是否连通"""
        if city_a not in self.graph or city_b not in self.graph:
            return False
        return nx.has_path(self.graph, city_a, city_b)
    
    def get_existing_segments(self):
        """获取所有已存在的线段"""
        segments = set()
        for a, b in self.graph.edges():
            normalized = tuple(sorted([a, b]))
            segments.add(normalized)
        return segments
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            "total_lines": len(self.lines),
            "total_stations": len(self.station_to_lines),
            "total_segments": len(self.graph.edges)
        }


def create_simple_topology():
    """创建默认的拓扑结构"""
    topology = SimpleHSRTopology()
    
    predefined_lines = [
        ("京哈高铁", ["北京", "沈阳", "长春", "哈尔滨"]),
        ("京沪高铁", ["北京", "天津", "济南", "南京", "上海"]),
        ("京港高铁", ["北京", "石家庄", "郑州", "武汉", "长沙", "广州", "深圳", "香港"]),
        ("沿海通道", ["大连", "沈阳", "天津", "青岛", "上海", "杭州", "宁波", "福州", "厦门", "深圳"]),
        ("京昆通道", ["北京", "石家庄", "太原", "西安", "成都", "昆明"]),
        ("陆桥通道", ["连云港", "徐州", "郑州", "西安", "兰州", "乌鲁木齐"]),
        ("沿江通道", ["上海", "南京", "合肥", "武汉", "重庆", "成都"]),
        ("沪昆通道", ["上海", "杭州", "南昌", "长沙", "贵阳", "昆明"]),
    ]
    
    for line_name, stations in predefined_lines:
        topology.add_line(line_name, stations)
    
    return topology

