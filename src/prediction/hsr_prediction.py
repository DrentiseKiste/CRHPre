# -*- coding: utf-8 -*-
"""
高铁预测程序 v2 - 优化版
核心优化：
1. 考虑线路途经的中间站 - 如果A、B都在已有线路的途经列表中，标记为冗余
2. 战略重要性加权 - 让长距离、大城市间的线路更容易被选中
3. 距离分段优化 - 不只看相邻城市，也关注300-800km"黄金距离"
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import math
import sys
import time
import json
import os

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# 添加项目根目录和路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_dir = os.path.join(project_root, "data")
output_dir = os.path.join(project_root, "output")

from china_hsr_prediction import CITIES_DATA

ADMIN_LEVEL_MAP = {
    "直辖市": 4, "特别行政区": 4, "省会/副省级": 3,
    "地级市": 2, "县级": 1
}

PROVINCE_MAP = {}
URBAN_AGGLOMERATIONS = {
    "长三角": ["上海", "南京", "杭州", "苏州", "无锡", "常州", "南通",
              "嘉兴", "湖州", "绍兴", "宁波", "舟山", "台州",
              "扬州", "镇江", "泰州", "盐城"],
    "珠三角": ["广州", "深圳", "珠海", "佛山", "惠州", "东莞",
              "中山", "江门", "肇庆", "香港", "澳门"],
    "京津冀": ["北京", "天津", "石家庄", "唐山", "秦皇岛",
              "邯郸", "邢台", "保定", "张家口", "承德",
              "沧州", "廊坊", "衡水"],
    "成渝": ["成都", "重庆", "自贡", "泸州", "德阳", "绵阳",
            "遂宁", "内江", "乐山", "南充", "眉山",
            "宜宾", "广安", "达州", "雅安"],
    "长江中游": ["武汉", "长沙", "南昌", "黄石", "襄阳", "宜昌",
                "衡阳", "株洲", "湘潭", "九江", "景德镇", "上饶"],
}

# 导入数据补充模块的覆盖信息
from hsr_data_enricher import HSRDataEnricher


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def count_connections(city_name, existing_pairs):
    count = 0
    for (a, b) in existing_pairs:
        if a == city_name or b == city_name:
            count += 1
    return count


def generate_features_fast(a, b, city_a_name, city_b_name, existing_pairs):
    """快速特征生成——接收预缓存的城市数据dict"""
    distance = haversine_distance(a['lat'], a['lon'], b['lat'], b['lon'])

    gdp_sum = a['gdp'] + b['gdp']
    gdp_product = a['gdp'] * b['gdp']
    pop_sum = a['population'] + b['population']
    pop_product = a['population'] * b['population']
    economic_gravity = (gdp_sum * pop_product) / (distance + 10)
    gdp_per_capita_a = a['gdp'] / (a['population'] + 1)
    gdp_per_capita_b = b['gdp'] / (b['population'] + 1)
    avg_gdp_per_capita = (gdp_per_capita_a + gdp_per_capita_b) / 2

    admin_max = max(a['admin_level_code'], b['admin_level_code'])
    admin_min = min(a['admin_level_code'], b['admin_level_code'])
    admin_gap = admin_max - admin_min

    is_golden = 1 if 300 <= distance <= 800 else 0
    is_medium = 1 if 200 <= distance <= 600 else 0
    is_short = 1 if distance < 200 else 0
    is_long = 1 if distance > 1000 else 0
    is_strategic = 1 if 300 <= distance <= 1200 else 0

    both_east = 1 if (a['east_of_hu'] and b['east_of_hu']) else 0
    one_east = 1 if (a['east_of_hu'] != b['east_of_hu']) else 0

    same_province = 0
    same_agglomeration = 0
    try:
        if a.get('province') and b.get('province') and a['province'] == b['province']:
            same_province = 1
        if a.get('agglomeration') and b.get('agglomeration') and a['agglomeration'] == b['agglomeration']:
            same_agglomeration = 1
    except:
        pass

    conn_a = count_connections(city_a_name, existing_pairs)
    conn_b = count_connections(city_b_name, existing_pairs)
    max_conn = max(conn_a, conn_b)
    sum_conn = conn_a + conn_b

    a_nb, b_nb = set(), set()
    for (x, y) in existing_pairs:
        if x == city_a_name: a_nb.add(y)
        if y == city_a_name: a_nb.add(x)
        if x == city_b_name: b_nb.add(y)
        if y == city_b_name: b_nb.add(x)
    common = len(a_nb.intersection(b_nb))

    is_cs = 0
    if city_a_name in ['福州', '厦门'] and city_b_name in ['台北', '台中', '高雄']:
        is_cs = 1
    if city_b_name in ['福州', '厦门'] and city_a_name in ['台北', '台中', '高雄']:
        is_cs = 1

    return {
        'gdp_sum': gdp_sum,
        'gdp_product': gdp_product,
        'pop_sum': pop_sum,
        'pop_product': pop_product,
        'economic_gravity': economic_gravity,
        'avg_gdp_per_capita': avg_gdp_per_capita,
        'distance_km': distance,
        'log_distance': math.log1p(distance),
        'is_golden_distance': is_golden,
        'is_medium_distance': is_medium,
        'is_short_distance': is_short,
        'is_long_distance': is_long,
        'is_strategic_distance': is_strategic,
        'admin_max': admin_max,
        'admin_min': admin_min,
        'admin_gap': admin_gap,
        'both_east': both_east,
        'one_east': one_east,
        'same_province': same_province,
        'same_agglomeration': same_agglomeration,
        'max_connections': max_conn,
        'sum_connections': sum_conn,
        'common_neighbors': common,
        'is_cross_sea': is_cs,
    }


def generate_features(cities_df, cities_dict, city_a_name, city_b_name, existing_pairs):
    """生成特征"""
    if city_a_name not in cities_df['name'].values or city_b_name not in cities_df['name'].values:
        return None

    a = cities_df[cities_df['name'] == city_a_name].iloc[0]
    b = cities_df[cities_df['name'] == city_b_name].iloc[0]

    distance = haversine_distance(a['lat'], a['lon'], b['lat'], b['lon'])

    gdp_sum = a['gdp'] + b['gdp']
    gdp_product = a['gdp'] * b['gdp']
    pop_sum = a['population'] + b['population']
    pop_product = a['population'] * b['population']
    economic_gravity = (gdp_sum * pop_product) / (distance + 10)
    gdp_per_capita_a = a['gdp'] / (a['population'] + 1)
    gdp_per_capita_b = b['gdp'] / (b['population'] + 1)
    avg_gdp_per_capita = (gdp_per_capita_a + gdp_per_capita_b) / 2

    admin_max = max(a['admin_level_code'], b['admin_level_code'])
    admin_min = min(a['admin_level_code'], b['admin_level_code'])
    admin_gap = admin_max - admin_min

    is_golden = 1 if 300 <= distance <= 800 else 0
    is_medium = 1 if 200 <= distance <= 600 else 0
    is_short = 1 if distance < 200 else 0
    is_long = 1 if distance > 1000 else 0
    is_strategic = 1 if 300 <= distance <= 1200 else 0  # 新增：战略距离

    both_east = 1 if (a['east_of_hu'] and b['east_of_hu']) else 0
    one_east = 1 if (a['east_of_hu'] != b['east_of_hu']) else 0

    try:
        same_prov = a.get('province') is not None and b.get('province') is not None and a.get('province') == b.get('province')
        sam_agg = a.get('agglomeration') is not None and b.get('agglomeration') is not None and a.get('agglomeration') == b.get('agglomeration')
        same_province = 1 if same_prov else 0
        same_agglomeration = 1 if sam_agg else 0
    except:
        same_province = 0
        same_agglomeration = 0

    connections_a = count_connections(city_a_name, existing_pairs)
    connections_b = count_connections(city_b_name, existing_pairs)
    max_connections = max(connections_a, connections_b)
    sum_connections = connections_a + connections_b

    common_neighbors = 0
    a_neighbors = set()
    b_neighbors = set()
    for (x, y) in existing_pairs:
        if x == city_a_name:
            a_neighbors.add(y)
        if y == city_a_name:
            a_neighbors.add(x)
        if x == city_b_name:
            b_neighbors.add(y)
        if y == city_b_name:
            b_neighbors.add(x)
    common_neighbors = len(a_neighbors.intersection(b_neighbors))

    is_cross_sea = 0
    if city_a_name in ['福州', '厦门'] and city_b_name in ['台北', '台中', '高雄']:
        is_cross_sea = 1
    if city_b_name in ['福州', '厦门'] and city_a_name in ['台北', '台中', '高雄']:
        is_cross_sea = 1

    features = {
        'gdp_sum': gdp_sum,
        'gdp_product': gdp_product,
        'pop_sum': pop_sum,
        'pop_product': pop_product,
        'economic_gravity': economic_gravity,
        'avg_gdp_per_capita': avg_gdp_per_capita,
        'distance_km': distance,
        'log_distance': math.log1p(distance),
        'is_golden_distance': is_golden,
        'is_medium_distance': is_medium,
        'is_short_distance': is_short,
        'is_long_distance': is_long,
        'is_strategic_distance': is_strategic,
        'admin_max': admin_max,
        'admin_min': admin_min,
        'admin_gap': admin_gap,
        'both_east': both_east,
        'one_east': one_east,
        'same_province': same_province,
        'same_agglomeration': same_agglomeration,
        'max_connections': max_connections,
        'sum_connections': sum_connections,
        'common_neighbors': common_neighbors,
        'is_cross_sea': is_cross_sea,
    }
    return features


class PredictionOptimizer:
    """预测优化器 - 三大优化"""

    def __init__(self):
        # 线路覆盖图（来自数据补充器）
        self.enricher = HSRDataEnricher()
        self.covered_pairs = self.enricher.get_existing_coverage()

        # 城市数据
        self.cities_dict = {row['name']: row for row in CITIES_DATA}

        print("="*60)
        print("预测优化器已初始化")
        print(f"  已有覆盖对: {len(self.covered_pairs)} 个")
        print(f"  途经线路:    {len(self.enricher.line_stations)} 条")
        print("="*60)

    def is_redundant_by_coverage(self, city_a, city_b):
        """
        优化1：检查两个城市是否已被已有线路覆盖
        如果A和B都在同一条已有线路的途经站点中（非相邻也覆盖），
        说明这两城市之间已有高铁连接，无需重复预测。

        例如：京沪高铁 北京→天津→济南→南京→上海
        覆盖了对：(北京,济南), (天津,南京), (北京,上海) 等
        """
        pair = tuple(sorted([city_a, city_b]))
        return pair in self.covered_pairs

    def calculate_strategic_score(self, city_a, city_b):
        """
        优化2：计算战略重要性得分
        让长距离、大城市之间的线路更容易被选中

        评分因素：
        - 行政级别（直辖市=4, 省会=3, 地级市=2, 县级=1）
        - 距离合适度（300-800km 最佳，800-1200km 次佳）
        - GDP 规模
        """
        if city_a not in self.cities_dict or city_b not in self.cities_dict:
            return 0

        a = self.cities_dict[city_a]
        b = self.cities_dict[city_b]
        dist = haversine_distance(a['lat'], a['lon'], b['lat'], b['lon'])

        # 行政级别分 (0~1) - 原始数据用admin_level字符串
        raw_admin = {'直辖市': 5, '特别行政区': 5, '省会/副省级': 4,
                     '地级市': 3, '县级': 2}
        admin_code_a = raw_admin.get(a.get('admin_level', ''), 2)
        admin_code_b = raw_admin.get(b.get('admin_level', ''), 2)
        admin_a = admin_code_a / 5.0
        admin_b = admin_code_b / 5.0
        admin_score = (admin_a + admin_b) / 2.0

        # 距离合适度 (0~1)
        if 300 <= dist <= 600:
            dist_score = 1.0       # 黄金距离
        elif 600 < dist <= 1000:
            dist_score = 0.9       # 很好的战略距离
        elif 1000 < dist <= 1500:
            dist_score = 0.7       # 可行长距离
        elif 150 < dist <= 300:
            dist_score = 0.5       # 较短
        else:
            dist_score = 0.2       # 太短或太长

        # GDP规模分 (0~1)
        gdp_a = min(a['gdp'] / 20000, 1.0)
        gdp_b = min(b['gdp'] / 20000, 1.0)
        gdp_score = (gdp_a + gdp_b) / 2.0

        # 综合战略分 = 35% 行政 + 35% 距离 + 30% GDP
        strategic = 0.35 * admin_score + 0.35 * dist_score + 0.30 * gdp_score

        return strategic

    def optimize_predictions(self, predictions, cities_dict):
        """
        优化3：综合排序
        70% 模型概率 + 30% 战略重要性

        同时标记被已有线路覆盖的冗余预测
        """
        optimized = []
        redundant = []

        for city_a, city_b, prob in predictions:
            # 检查冗余
            if self.is_redundant_by_coverage(city_a, city_b):
                redundant.append((city_a, city_b, prob))
                continue

            # 计算战略得分
            strategic = self.calculate_strategic_score(city_a, city_b)

            # 综合得分 = 70% 模型 + 30% 战略
            combined = 0.70 * prob + 0.30 * strategic

            optimized.append((city_a, city_b, prob, strategic, combined))

        # 按综合得分排序
        optimized.sort(key=lambda x: x[4], reverse=True)

        return optimized, redundant


def build_city_dataframe(cities_data):
    df = pd.DataFrame(cities_data)
    df['admin_level_code'] = df['admin_level'].map(ADMIN_LEVEL_MAP)
    df['east_of_hu'] = (df['lon'] > 120) | ((df['lon'] > 100) & (df['lat'] > 35))
    df['province'] = None
    df['agglomeration'] = None
    for agg_name, cities in URBAN_AGGLOMERATIONS.items():
        df.loc[df['name'].isin(cities), 'agglomeration'] = agg_name
    return df


def rolling_prediction_v2(cities_df, history_df, optimizer,
                           start_year=2026, end_year=2030, prob_threshold=0.88):
    """滚动预测 v2 - 优化版"""
    print("\n" + "="*60)
    print("开始滚动预测 (v2 优化版)")
    print("="*60)

    cities_dict = {row['name']: row for _, row in cities_df.iterrows()}
    city_names = set(cities_df['name'])

    cumulative_routes = []
    for idx, row in history_df.iterrows():
        year = int(row['open_year'])
        if year <= 2025:
            start = str(row['start_city']).strip()
            end = str(row['end_city']).strip()
            if start in city_names and end in city_names:
                cumulative_routes.append((start, end))

    print(f"初始累积线路: {len(cumulative_routes)}条")

    all_predictions = {}
    all_redundant = {}
    all_cities = cities_df['name'].tolist()

    # 过滤掉已经覆盖了太多的城市对
    # 如果一对城市已经被已有线路覆盖，就从候选列表中移除
    covered_set = set(optimizer.covered_pairs.keys())

    for year in range(start_year, end_year + 1):
        t0 = time.time()
        print(f"\n[{year}年] 预测中... ({len(cumulative_routes)}条基础线路)")

        existing_pairs = set()
        for city_a, city_b in cumulative_routes:
            existing_pairs.add(tuple(sorted([city_a, city_b])))

        # 正样本
        np.random.seed(year)
        if len(cumulative_routes) > 80:
            idx_sample = list(np.random.choice(len(cumulative_routes), 80, replace=False))
            positive_pairs = [cumulative_routes[i] for i in idx_sample]
        else:
            positive_pairs = cumulative_routes[:]

        # 负样本
        negative_pairs = []
        attempts = 0
        while len(negative_pairs) < len(positive_pairs) * 3 and attempts < 10000:
            attempts += 1
            ca = np.random.choice(all_cities)
            cb = np.random.choice(all_cities)
            if ca == cb:
                continue
            pair = tuple(sorted([ca, cb]))
            if pair not in existing_pairs and pair not in negative_pairs:
                negative_pairs.append(pair)

        X_train, y_train = [], []
        for ca, cb in positive_pairs:
            feat = generate_features(cities_df, cities_dict, ca, cb, existing_pairs)
            if feat:
                X_train.append(feat)
                y_train.append(1)
        for ca, cb in negative_pairs:
            feat = generate_features(cities_df, cities_dict, ca, cb, existing_pairs)
            if feat:
                X_train.append(feat)
                y_train.append(0)

        if not X_train:
            continue

        X_df = pd.DataFrame(X_train)
        y_array = np.array(y_train)

        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.15,
                random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
        else:
            model = RandomForestClassifier(
                n_estimators=80, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_df, y_array)

        # 构建候选对 + 特征（快速版——预计算城市特征缓存）
        city_cache = {}
        for _, row in cities_df.iterrows():
            city_cache[row['name']] = row.to_dict()

        candidates = []
        X_pred = []
        zero_feat = {k: 0 for k in X_df.columns}

        for i in range(len(all_cities)):
            ca = all_cities[i]
            ca_cache = city_cache.get(ca)
            for j in range(i + 1, len(all_cities)):
                cb = all_cities[j]
                pair = tuple(sorted([ca, cb]))
                if pair in existing_pairs:
                    continue
                candidates.append((ca, cb))
                # 快速特征生成（使用缓存）
                cb_cache = city_cache.get(cb)
                if ca_cache and cb_cache:
                    feat = generate_features_fast(
                        ca_cache, cb_cache, ca, cb, existing_pairs)
                    X_pred.append(feat if feat else zero_feat)
                else:
                    X_pred.append(zero_feat)

        X_pred_df = pd.DataFrame(X_pred)
        probas = model.predict_proba(X_pred_df)[:, 1]

        # 收集原始预测
        raw_predictions = []
        for (ca, cb), prob in zip(candidates, probas):
            if prob >= prob_threshold:
                raw_predictions.append((ca, cb, prob))

        raw_predictions.sort(key=lambda x: x[2], reverse=True)

        # 应用优化器
        optimized, redundant = optimizer.optimize_predictions(
            raw_predictions, cities_dict)

        # Top 限制以避免爆炸
        max_new = 30
        optimized = optimized[:max_new]
        redundant = redundant[:max_new * 2]

        t1 = time.time()
        print(f"  原始预测: {len(raw_predictions)}条")
        print(f"  过滤冗余: {len(redundant)}条")
        print(f"  最终保留: {len(optimized)}条")
        print(f"  耗时: {t1-t0:.1f}s")

        if optimized:
            print("  Top 8:")
            for ca, cb, prob, strat, comb in optimized[:8]:
                dist = haversine_distance(
                    cities_dict[ca]['lat'], cities_dict[ca]['lon'],
                    cities_dict[cb]['lat'], cities_dict[cb]['lon'])
                print(f"    {ca}-{cb}: 综合{comb:.3f} "
                      f"(模型{prob:.3f}+战略{strat:.3f}) {dist:.0f}km")

        # 添加到累积线路
        for ca, cb, _, _, _ in optimized:
            cumulative_routes.append((ca, cb))

        # 更新覆盖集
        for ca, cb, _, _, _ in optimized:
            pair = tuple(sorted([ca, cb]))
            covered_set.add(pair)

        all_predictions[year] = optimized
        all_redundant[year] = redundant

    return all_predictions, all_redundant


def time_split_validation_v2(cities_df, history_df, optimizer):
    """时间分割验证"""
    print("\n" + "="*60)
    print("【时间分割验证】")
    print("="*60)

    cities_dict = {row['name']: row for _, row in cities_df.iterrows()}
    city_names = set(cities_df['name'])

    # 训练: 2008-2020
    train_pairs = set()
    for _, row in history_df.iterrows():
        year = int(row['open_year'])
        if year <= 2020:
            s, e = str(row['start_city']).strip(), str(row['end_city']).strip()
            if s in city_names and e in city_names:
                train_pairs.add(tuple(sorted([s, e])))

    # 验证: 2021-2025
    val_pairs = set()
    for _, row in history_df.iterrows():
        year = int(row['open_year'])
        if 2021 <= year <= 2025:
            s, e = str(row['start_city']).strip(), str(row['end_city']).strip()
            if s in city_names and e in city_names:
                p = tuple(sorted([s, e]))
                if p not in train_pairs:
                    val_pairs.add(p)

    X_train, y_train = [], []
    for a, b in train_pairs:
        feat = generate_features(cities_df, cities_dict, a, b, train_pairs)
        if feat:
            X_train.append(feat)
            y_train.append(1)

    all_list = list(city_names)
    np.random.seed(42)
    neg = 0
    while neg < len(train_pairs) * 3:
        a = np.random.choice(all_list)
        b = np.random.choice(all_list)
        if a == b:
            continue
        p = tuple(sorted([a, b]))
        if p not in train_pairs and p not in val_pairs:
            feat = generate_features(cities_df, cities_dict, a, b, train_pairs)
            if feat:
                X_train.append(feat)
                y_train.append(0)
                neg += 1

    X_df = pd.DataFrame(X_train)
    y_arr = np.array(y_train)

    if XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.15,
            random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
    else:
        model = RandomForestClassifier(
            n_estimators=80, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_df, y_arr)

    X_val, y_val = [], []
    for a, b in val_pairs:
        feat = generate_features(cities_df, cities_dict, a, b, train_pairs)
        if feat:
            X_val.append(feat)
            y_val.append(1)

    neg = 0
    while neg < len(val_pairs) * 3:
        a = np.random.choice(all_list)
        b = np.random.choice(all_list)
        if a == b:
            continue
        p = tuple(sorted([a, b]))
        if p not in train_pairs and p not in val_pairs:
            feat = generate_features(cities_df, cities_dict, a, b, train_pairs)
            if feat:
                X_val.append(feat)
                y_val.append(0)
                neg += 1

    Xv = pd.DataFrame(X_val)
    y_prob = model.predict_proba(Xv)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)

    print(f"AUC-ROC: {auc:.4f}")
    print(f"准确率: {acc:.4f}")

    # 检查优化器剔除的冗余
    redundant_count = 0
    for a, b in val_pairs:
        if optimizer.is_redundant_by_coverage(a, b):
            redundant_count += 1
    print(f"验证集中的覆盖冗余: {redundant_count}/{len(val_pairs)}")

    return {"auc": auc, "accuracy": acc, "redundant": redundant_count}


def save_results(all_predictions, all_redundant, output=None):
    """保存预测结果"""
    if output is None:
        output = os.path.join(output_dir, "predicted_routes_v2.csv")

    rows = []
    for year, preds in all_predictions.items():
        for ca, cb, prob, strat, comb in preds:
            rows.append({
                "year": year, "city_a": ca, "city_b": cb,
                "probability": prob, "strategic_score": strat,
                "combined_score": comb
            })
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n预测结果已保存: {output} ({len(df)}条)")

    # 冗余报告
    red_rows = []
    for year, reds in all_redundant.items():
        for ca, cb, prob in reds:
            red_rows.append({
                "year": year, "city_a": ca, "city_b": cb,
                "probability": prob
            })
    if red_rows:
        red_path = os.path.join(output_dir, "redundant_routes_v2.csv")
        red_df = pd.DataFrame(red_rows)
        red_df.to_csv(red_path, index=False, encoding="utf-8-sig")
        print(f"冗余预测已保存: {red_path} ({len(red_df)}条)")

    return df


def main():
    print("="*60)
    print("高铁预测系统 v2 - 优化版")
    print("="*60)

    cities_df = build_city_dataframe(CITIES_DATA)
    history_df = pd.read_csv(os.path.join(data_dir, "hsr_history.csv"), encoding="utf-8-sig")
    print(f"加载 {len(history_df)} 条历史线路")

    # 初始化优化器
    optimizer = PredictionOptimizer()

    # 验证
    val_result = time_split_validation_v2(cities_df, history_df, optimizer)

    # 预测
    all_preds, all_reds = rolling_prediction_v2(
        cities_df, history_df, optimizer,
        start_year=2026, end_year=2030, prob_threshold=0.88)

    # 保存
    df = save_results(all_preds, all_reds)

    # 报告
    print("\n" + "="*60)
    print("预测报告")
    print("="*60)
    stats = df.groupby('year').size()
    for yr, cnt in stats.items():
        print(f"  {yr}年: {cnt}条")

    if len(df) > 0:
        print(f"\n  平均距离: {df['probability'].mean():.3f}")

    print("\n完成!")


if __name__ == "__main__":
    main()