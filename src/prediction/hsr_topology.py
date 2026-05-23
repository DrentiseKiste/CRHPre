
# -*- coding: utf-8 -*-
"""
高铁线路拓扑结构模块
用于构建和管理高铁线路的完整拓扑关系
"""

from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import networkx as nx


class LineLevel(Enum):
    """线路层级"""
    TRUNK = "trunk"
    BRANCH = "branch"
    LOCAL = "local"


@dataclass
class HSRLine:
    """高铁线路数据结构"""
    line_name: str
    stations: List[str]
    level: LineLevel = LineLevel.TRUNK
    open_year: Optional[int] = None
    length_km: Optional[float] = None
    max_speed: Optional[int] = None
    
    def __post_init__(self):
        if len(self.stations) &lt; 2:
            raise ValueError("线路至少需要两个站点")
    
    def get_all_segments(self):
        """获取线路的所有相邻线段"""
        return [(self.stations[i], self.stations[i+1]) 
                for i in range(len(self.stations)-1)]
    
    def get_all_city_pairs(self):
        """获取线路中所有可能的城市对"""
        pairs = []
        for i in range(len(self.stations)):
            for j in range(i+1, len(self.stations)):
                pairs.append((self.stations[i], self.stations[j]))
        return pairs
    
    def contains_path(self, city_a, city_b):
        """检查该线路是否包含从city_a到city_b的完整路径"""
        try:
            self.stations.index(city_a)
            self.stations.index(city_b)
            return True
        except ValueError:
            return False
    
    def is_direct_segment(self, city_a, city_b):
        """检查两个城市是否是该线路上的直接相邻段"""
        segments = self.get_all_segments()
        return (city_a, city_b) in segments or (city_b, city_a) in segments


@dataclass
class TopologyConflict:
    """拓扑冲突信息"""
    conflict_type: str
    candidate_pair: Tuple[str, str]
    existing_line: str
    description: str
    severity: int = 1


class HSRTopology:
    """高铁线路拓扑结构管理器"""
    
    def __init__(self):
        self.lines = {}
        self.station_to_lines = defaultdict(set)
        self.graph = nx.Graph()
        self._build_complete = False
    
    def add_line(self, line):
        """添加一条高铁线路"""
        self.lines[line.line_name] = line
        
        for station in line.stations:
            self.station_to_lines[station].add(line.line_name)
        
        for a, b in line.get_all_segments():
            self.graph.add_edge(a, b, line=line.line_name)
    
    def build_lines_from_history(self, history_data):
        """从历史数据构建完整线路拓扑"""
        line_groups = defaultdict(list)
        for item in history_data:
            line_name = item.get("line_name", "Unknown")
            line_groups[line_name].append(item)
        
        for line_name, segments in line_groups.items():
            if len(segments) == 1:
                seg = segments[0]
                line = HSRLine(
                    line_name=line_name,
                    stations=[seg["start_city"], seg["end_city"]],
                    open_year=seg.get("open_year"),
                    length_km=seg.get("length_km"),
                    max_speed=seg.get("max_speed")
                )
                self.add_line(line)
            else:
                full_stations = self._build_full_path(segments)
                if full_stations:
                    line = HSRLine(
                        line_name=line_name,
                        stations=full_stations,
                        open_year=segments[0].get("open_year")
                    )
                    self.add_line(line)
    
    def _build_full_path(self, segments):
        """从多个线段构建完整路径"""
        temp_graph = nx.Graph()
        for seg in segments:
            temp_graph.add_edge(seg["start_city"], seg["end_city"])
        
        if not nx.is_connected(temp_graph):
            return None
        
        endpoints = [node for node, degree in temp_graph.degree() if degree == 1]
        
        if len(endpoints) != 2:
            return list(temp_graph.nodes())
        
        try:
            path = nx.shortest_path(temp_graph, endpoints[0], endpoints[1])
            return path
        except:
            return None
    
    def get_station_lines(self, station):
        """获取一个站点所属的所有线路"""
        line_names = self.station_to_lines.get(station, set())
        return [self.lines[name] for name in line_names]
    
    def find_connecting_lines(self, city_a, city_b):
        """查找连接两个城市的所有线路"""
        common_lines = set()
        if city_a in self.station_to_lines and city_b in self.station_to_lines:
            common_lines = self.station_to_lines[city_a] &amp; self.station_to_lines[city_b]
        
        valid_lines = []
        for line_name in common_lines:
            line = self.lines[line_name]
            if line.contains_path(city_a, city_b):
                valid_lines.append(line)
        
        return valid_lines
    
    def check_conflicts(self, city_a, city_b):
        """检查预测线路是否存在拓扑冲突"""
        conflicts = []
        pair = tuple(sorted([city_a, city_b]))
        
        for line in self.lines.values():
            if line.is_direct_segment(city_a, city_b):
                conflicts.append(TopologyConflict(
                    conflict_type="redundant_segment",
                    candidate_pair=pair,
                    existing_line=line.line_name,
                    description="线路 %s 已包含直接线段 %s-%s" % (line.line_name, city_a, city_b),
                    severity=5
                ))
        
        connecting_lines = self.find_connecting_lines(city_a, city_b)
        for line in connecting_lines:
            if not line.is_direct_segment(city_a, city_b):
                conflicts.append(TopologyConflict(
                    conflict_type="existing_path",
                    candidate_pair=pair,
                    existing_line=line.line_name,
                    description="线路 %s 已覆盖 %s 到 %s 的完整路径" % (line.line_name, city_a, city_b),
                    severity=3
                ))
        
        parallel_conflicts = self._check_parallel_lines(city_a, city_b)
        conflicts.extend(parallel_conflicts)
        
        return conflicts
    
    def _check_parallel_lines(self, city_a, city_b):
        """检查是否与现有线路过于平行"""
        conflicts = []
        return conflicts
    
    def is_pair_connected(self, city_a, city_b):
        """检查两个城市在当前网络中是否已经连通"""
        if city_a not in self.graph or city_b not in self.graph:
            return False
        return nx.has_path(self.graph, city_a, city_b)
    
    def get_existing_segments(self):
        """获取所有已存在的线段"""
        segments = set()
        for line in self.lines.values():
            for seg in line.get_all_segments():
                normalized = tuple(sorted(seg))
                segments.add(normalized)
        return segments
    
    def get_statistics(self):
        """获取拓扑统计信息"""
        return {
            "total_lines": len(self.lines),
            "total_stations": len(self.station_to_lines),
            "total_segments": len(self.graph.edges),
            "connected_components": nx.number_connected_components(self.graph)
        }


def create_default_topology(history_data=None):
    """创建默认的拓扑结构"""
    topology = HSRTopology()
    
    predefined_lines = [
        {
            "line_name": "京哈高铁",
            "stations": ["北京", "沈阳", "长春", "哈尔滨"],
            "level": LineLevel.TRUNK
        },
        {
            "line_name": "京沪高铁",
            "stations": ["北京", "天津", "济南", "南京", "上海"],
            "level": LineLevel.TRUNK
        },
        {
            "line_name": "京港高铁",
            "stations": ["北京", "石家庄", "郑州", "武汉", "长沙", "广州", "深圳", "香港"],
            "level": LineLevel.TRUNK
        },
        {
            "line_name": "沿海通道",
            "stations": ["大连", "沈阳", "天津", "青岛", "上海", "杭州", "宁波", "福州", "厦门", "深圳"],
            "level": LineLevel.TRUNK
        },
        {
            "line_name": "京昆通道",
            "stations": ["北京", "石家庄", "太原", "西安", "成都", "昆明"],
            "level": LineLevel.TRUNK
        },
        {
            "line_name": "陆桥通道",
            "stations": ["连云港", "徐州", "郑州", "西安", "兰州", "乌鲁木齐"],
            "level": LineLevel.TRUNK
        },
        {
            "line_name": "沿江通道",
            "stations": ["上海", "南京", "合肥", "武汉", "重庆", "成都"],
            "level": LineLevel.TRUNK
        },
        {
            "line_name": "沪昆通道",
            "stations": ["上海", "杭州", "南昌", "长沙", "贵阳", "昆明"],
            "level": LineLevel.TRUNK
        },
    ]
    
    for line_data in predefined_lines:
        line = HSRLine(
            line_name=line_data["line_name"],
            stations=line_data["stations"],
            level=line_data.get("level", LineLevel.TRUNK)
        )
        topology.add_line(line)
    
    if history_data:
        topology.build_lines_from_history(history_data)
    
    return topology

