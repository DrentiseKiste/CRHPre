# -*- coding: utf-8 -*-
"""
中国高铁滚动预测系统 - 增强版
每年以前面的所有线路（历史+预测）为基础，预测新建线路
增加功能：空间拓扑修正、更多特征、性能优化
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import sys
import time

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[警告] sklearn 未安装")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[警告] xgboost 未安装")

from china_hsr_prediction import CITIES_DATA

ADMIN_LEVEL_MAP = {
    "直辖市": 4, "特别行政区": 4, "省会/副省级": 3,
    "地级市": 2, "县级": 1
}

URBAN_AGGLOMERATIONS = {
    "长三角": ["上海", "南京", "杭州", "苏州", "无锡", "常州", "南通", "嘉兴", "湖州", "绍兴", "宁波", "舟山", "台州", "扬州", "镇江", "泰州", "盐城"],
    "珠三角": ["广州", "深圳", "珠海", "佛山", "惠州", "东莞", "中山", "江门", "肇庆", "香港", "澳门"],
    "京津冀": ["北京", "天津", "石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定", "张家口", "承德", "沧州", "廊坊", "衡水"],
    "成渝": ["成都", "重庆", "自贡", "泸州", "德阳", "绵阳", "遂宁", "内江", "乐山", "南充", "眉山", "宜宾", "广安", "达州", "雅安"],
    "长江中游": ["武汉", "长沙", "南昌", "黄石", "襄阳", "宜昌", "衡阳", "株洲", "湘潭", "九江", "景德镇", "上饶"],
}

PROVINCE_MAP = {
    "北京": "北京", "天津": "天津", "上海": "上海", "重庆": "重庆",
    "香港": "香港", "澳门": "澳门", "台北": "台湾",
    "哈尔滨": "黑龙江", "长春": "吉林", "沈阳": "辽宁", "大连": "辽宁",
    "石家庄": "河北", "太原": "山西", "呼和浩特": "内蒙古",
    "济南": "山东", "青岛": "山东", "南京": "江苏", "苏州": "江苏", "无锡": "江苏", "徐州": "江苏", "常州": "江苏", "南通": "江苏", "扬州": "江苏", "镇江": "江苏", "泰州": "江苏", "盐城": "江苏", "连云港": "江苏",
    "杭州": "浙江", "宁波": "浙江", "温州": "浙江", "绍兴": "浙江", "嘉兴": "浙江", "湖州": "浙江", "台州": "浙江", "金华": "浙江", "丽水": "浙江", "衢州": "浙江", "舟山": "浙江",
    "合肥": "安徽", "芜湖": "安徽", "马鞍山": "安徽", "安庆": "安徽", "蚌埠": "安徽", "阜阳": "安徽", "宿州": "安徽", "滁州": "安徽",
    "福州": "福建", "厦门": "福建", "泉州": "福建", "漳州": "福建", "莆田": "福建", "龙岩": "福建", "三明": "福建", "南平": "福建", "宁德": "福建",
    "南昌": "江西", "九江": "江西", "赣州": "江西", "上饶": "江西", "景德镇": "江西", "萍乡": "江西",
    "郑州": "河南", "洛阳": "河南", "开封": "河南", "新乡": "河南", "安阳": "河南", "焦作": "河南", "许昌": "河南", "商丘": "河南", "周口": "河南",
    "武汉": "湖北", "宜昌": "湖北", "襄阳": "湖北", "荆州": "湖北", "黄石": "湖北", "十堰": "湖北", "孝感": "湖北", "荆门": "湖北",
    "长沙": "湖南", "衡阳": "湖南", "株洲": "湖南", "湘潭": "湖南", "岳阳": "湖南", "常德": "湖南", "郴州": "湖南", "永州": "湖南", "邵阳": "湖南",
    "广州": "广东", "深圳": "广东", "珠海": "广东", "佛山": "广东", "东莞": "广东", "中山": "广东", "惠州": "广东", "江门": "广东", "肇庆": "广东", "汕头": "广东", "潮州": "广东", "揭阳": "广东", "汕尾": "广东", "湛江": "广东", "茂名": "广东", "阳江": "广东", "云浮": "广东", "韶关": "广东", "清远": "广东",
    "南宁": "广西", "柳州": "广西", "桂林": "广西", "梧州": "广西", "北海": "广西", "钦州": "广西", "贵港": "广西", "玉林": "广西", "百色": "广西",
    "海口": "海南", "三亚": "海南",
    "成都": "四川", "绵阳": "四川", "德阳": "四川", "宜宾": "四川", "泸州": "四川", "南充": "四川", "达州": "四川", "乐山": "四川", "自贡": "四川", "内江": "四川", "遂宁": "四川", "广安": "四川", "资阳": "四川", "眉山": "四川", "雅安": "四川", "巴中": "四川",
    "贵阳": "贵州", "遵义": "贵州", "六盘水": "贵州", "安顺": "贵州",
    "昆明": "云南", "曲靖": "云南", "玉溪": "云南",
    "拉萨": "西藏",
    "西安": "陕西", "宝鸡": "陕西", "咸阳": "陕西", "渭南": "陕西", "汉中": "陕西", "安康": "陕西", "商洛": "陕西", "延安": "陕西", "榆林": "陕西", "铜川": "陕西",
    "兰州": "甘肃", "天水": "甘肃", "酒泉": "甘肃", "嘉峪关": "甘肃", "张掖": "甘肃", "武威": "甘肃", "白银": "甘肃", "定西": "甘肃", "庆阳": "甘肃", "平凉": "甘肃", "陇南": "甘肃",
    "西宁": "青海",
    "银川": "宁夏",
    "乌鲁木齐": "新疆", "哈密": "新疆", "吐鲁番": "新疆", "克拉玛依": "新疆",
}


def build_city_dataframe(cities_data):
    """构建城市数据"""
    df = pd.DataFrame(cities_data)
    df['admin_level_code'] = df['admin_level'].map(ADMIN_LEVEL_MAP)
    
    # 添加省份列
    df['province'] = df['name'].map(PROVINCE_MAP)
    
    # 添加城市群列
    df['agglomeration'] = None
    for agg, cities in URBAN_AGGLOMERATIONS.items():
        df.loc[df['name'].isin(cities), 'agglomeration'] = agg
    
    # 添加胡焕庸线判断
    df['east_of_hu'] = (df['lon'] > 120) | ((df['lon'] > 100) & (df['lat'] > 35))
    
    return df


def load_history_data(csv_path="hsr_history.csv"):
    """加载历史高铁数据"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df['open_year'] = pd.to_numeric(df['open_year'], errors='coerce').fillna(0).astype(int)
        print("[历史数据] 加载了", len(df), "条线路记录")
        return df
    except FileNotFoundError:
        print("[警告] 未找到", csv_path)
        return pd.DataFrame()


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算Haversine距离"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_angle(lat1, lon1, lat2, lon2, lat3, lon3):
    """计算两条线路的夹角（度）
    线路1: (lat1, lon1) → (lat2, lon2)
    线路2: (lat1, lon1) → (lat3, lon3)
    """
    dx1 = lon2 - lon1
    dy1 = lat2 - lat1
    dx2 = lon3 - lon1
    dy2 = lat3 - lat1
    
    dot = dx1 * dx2 + dy1 * dy2
    mag1 = math.sqrt(dx1**2 + dy1**2)
    mag2 = math.sqrt(dx2**2 + dy2**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1, min(1, cos_theta))
    theta = math.acos(cos_theta)
    return math.degrees(theta)


def check_overlap(candidate_a, candidate_b, existing_routes, cities_dict, 
                  min_angle_threshold=30, min_distance_ratio=0.5):
    """
    检查预测线路是否与已有线路重叠
    返回 True 表示重叠，应该放弃
    """
    try:
        ca_lat = cities_dict[candidate_a]['lat']
        ca_lon = cities_dict[candidate_a]['lon']
        cb_lat = cities_dict[candidate_b]['lat']
        cb_lon = cities_dict[candidate_b]['lon']
    except KeyError:
        return False
    
    candidate_dist = haversine_distance(ca_lat, ca_lon, cb_lat, cb_lon)
    if candidate_dist == 0:
        return False
    
    for (city_x, city_y) in existing_routes:
        try:
            cx_lat = cities_dict[city_x]['lat']
            cx_lon = cities_dict[city_x]['lon']
            cy_lat = cities_dict[city_y]['lat']
            cy_lon = cities_dict[city_y]['lon']
        except KeyError:
            continue
        
        existing_dist = haversine_distance(cx_lat, cx_lon, cy_lat, cy_lon)
        
        # 检查是否共享一个端点
        if candidate_a == city_x:
            angle = calculate_angle(cx_lat, cx_lon, cy_lat, cy_lon, cb_lat, cb_lon)
            if angle < min_angle_threshold:
                return True
        elif candidate_a == city_y:
            angle = calculate_angle(cy_lat, cy_lon, cx_lat, cx_lon, cb_lat, cb_lon)
            if angle < min_angle_threshold:
                return True
        elif candidate_b == city_x:
            angle = calculate_angle(cx_lat, cx_lon, cy_lat, cy_lon, ca_lat, ca_lon)
            if angle < min_angle_threshold:
                return True
        elif candidate_b == city_y:
            angle = calculate_angle(cy_lat, cy_lon, cx_lat, cx_lon, ca_lat, ca_lon)
            if angle < min_angle_threshold:
                return True
        
        # 检查线路端点是否都在现有线路附近（接近平行）
        dist_ac = haversine_distance(ca_lat, ca_lon, cx_lat, cx_lon)
        dist_ad = haversine_distance(ca_lat, ca_lon, cy_lat, cy_lon)
        dist_bc = haversine_distance(cb_lat, cb_lon, cx_lat, cx_lon)
        dist_bd = haversine_distance(cb_lat, cb_lon, cy_lat, cy_lon)
        
        min_dist = min(dist_ac + dist_bd, dist_ad + dist_bc)
        if min_dist < candidate_dist * min_distance_ratio:
            return True
    
    return False


def count_connections(city_name, existing_pairs):
    """计算城市已有连接数"""
    count = 0
    for (a, b) in existing_pairs:
        if a == city_name or b == city_name:
            count += 1
    return count


def generate_enhanced_features(cities_df, cities_dict, city_a_name, city_b_name, existing_pairs):
    """生成增强版特征"""
    if city_a_name not in cities_df['name'].values or city_b_name not in cities_df['name'].values:
        return None
    
    a = cities_df[cities_df['name'] == city_a_name].iloc[0]
    b = cities_df[cities_df['name'] == city_b_name].iloc[0]
    
    distance = haversine_distance(a['lat'], a['lon'], b['lat'], b['lon'])
    
    # 经济特征
    gdp_sum = a['gdp'] + b['gdp']
    gdp_product = a['gdp'] * b['gdp']
    pop_sum = a['population'] + b['population']
    pop_product = a['population'] * b['population']
    economic_gravity = (gdp_sum * pop_product) / (distance + 10)
    gdp_per_capita_a = a['gdp'] / (a['population'] + 1)
    gdp_per_capita_b = b['gdp'] / (b['population'] + 1)
    avg_gdp_per_capita = (gdp_per_capita_a + gdp_per_capita_b) / 2
    
    # 行政特征
    admin_max = max(a['admin_level_code'], b['admin_level_code'])
    admin_min = min(a['admin_level_code'], b['admin_level_code'])
    admin_gap = admin_max - admin_min
    
    # 空间特征
    is_golden = 1 if 300 <= distance <= 800 else 0
    is_short = 1 if distance < 200 else 0
    is_long = 1 if distance > 1000 else 0
    
    # 地理特征
    both_east = 1 if (a['east_of_hu'] and b['east_of_hu']) else 0
    one_east = 1 if (a['east_of_hu'] != b['east_of_hu']) else 0
    same_province = 1 if (a['province'] is not None and b['province'] is not None and a['province'] == b['province']) else 0
    same_agglomeration = 1 if (a['agglomeration'] is not None and b['agglomeration'] is not None and a['agglomeration'] == b['agglomeration']) else 0
    
    # 网络特征
    connections_a = count_connections(city_a_name, existing_pairs)
    connections_b = count_connections(city_b_name, existing_pairs)
    max_connections = max(connections_a, connections_b)
    sum_connections = connections_a + connections_b
    
    # 计算共同邻居数
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
    
    # 跨海通道检测（简化）
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
        'is_short_distance': is_short,
        'is_long_distance': is_long,
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


def rolling_prediction(cities_df, history_df, start_year=2026, end_year=2030, prob_threshold=0.90):
    """滚动预测 - 增强版"""
    print("\n" + "="*60)
    print("开始滚动预测（增强版）")
    print("="*60)
    print(f"预测年份范围: {start_year}-{end_year}")
    print(f"概率阈值: {prob_threshold}")
    print(f"空间拓扑修正: 启用（夹角<30度过滤）")
    print(f"特征维度: 增强版（22个特征）")
    sys.stdout.flush()
    
    start_time_total = time.time()
    
    cities_dict = {row['name']: row for _, row in cities_df.iterrows()}
    city_names = set(cities_df['name'])
    
    # 获取2025年及之前的所有历史线路
    cumulative_routes = []
    for idx, row in history_df.iterrows():
        year = int(row['open_year'])
        if year <= 2025:
            start = str(row['start_city']).strip()
            end = str(row['end_city']).strip()
            if start in city_names and end in city_names:
                cumulative_routes.append((start, end))
    
    print(f"初始累积线路: {len(cumulative_routes)}条")
    sys.stdout.flush()
    
    all_predictions = {}
    all_cities = cities_df['name'].tolist()
    
    for year in range(start_year, end_year + 1):
        year_start = time.time()
        
        print(f"\n[{year}年] 开始预测...")
        print(f"    基础线路: {len(cumulative_routes)}条")
        sys.stdout.flush()
        
        # 构建已有线路集合
        existing_pairs = set()
        for city_a, city_b in cumulative_routes:
            existing_pairs.add(tuple(sorted([city_a, city_b])))
        
        # 正样本：随机选取最多80条
        np.random.seed(year)
        if len(cumulative_routes) > 80:
            positive_sample = list(np.random.choice(len(cumulative_routes), 80, replace=False))
            positive_pairs = [cumulative_routes[i] for i in positive_sample]
        else:
            positive_pairs = cumulative_routes[:]
        
        # 负样本：数量是正样本的3倍
        negative_pairs = []
        attempts = 0
        max_attempts = len(positive_pairs) * 40
        while len(negative_pairs) < len(positive_pairs) * 3 and attempts < max_attempts:
            attempts += 1
            city_a = np.random.choice(all_cities)
            city_b = np.random.choice(all_cities)
            if city_a == city_b:
                continue
            pair = tuple(sorted([city_a, city_b]))
            if pair not in existing_pairs and pair not in negative_pairs:
                negative_pairs.append(pair)
        
        print(f"    正样本: {len(positive_pairs)}条, 负样本: {len(negative_pairs)}条")
        sys.stdout.flush()
        
        # 生成训练数据（增强版特征）
        X_train = []
        y_train = []
        
        for city_a, city_b in positive_pairs:
            feat = generate_enhanced_features(cities_df, cities_dict, city_a, city_b, existing_pairs)
            if feat:
                X_train.append(feat)
                y_train.append(1)
        
        for city_a, city_b in negative_pairs:
            feat = generate_enhanced_features(cities_df, cities_dict, city_a, city_b, existing_pairs)
            if feat:
                X_train.append(feat)
                y_train.append(0)
        
        if not X_train:
            print("    [警告] 没有训练数据")
            continue
        
        X_df = pd.DataFrame(X_train)
        y_array = np.array(y_train)
        
        # 训练模型（精简参数，提升速度）
        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.15,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_df, y_array)
        else:
            model = RandomForestClassifier(
                n_estimators=80,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_df, y_array)
        
        # 预测所有未连接的城市对
        candidates = []
        for i in range(len(all_cities)):
            for j in range(i + 1, len(all_cities)):
                city_a = all_cities[i]
                city_b = all_cities[j]
                pair = tuple(sorted([city_a, city_b]))
                if pair not in existing_pairs:
                    candidates.append((city_a, city_b))
        
        print(f"    待预测城市对: {len(candidates)}条")
        sys.stdout.flush()
        
        # 批量预测
        X_pred = []
        for city_a, city_b in candidates:
            feat = generate_enhanced_features(cities_df, cities_dict, city_a, city_b, existing_pairs)
            if feat:
                X_pred.append(feat)
            else:
                X_pred.append({k: 0 for k in X_df.columns})
        
        X_pred_df = pd.DataFrame(X_pred)
        probas = model.predict_proba(X_pred_df)[:, 1]
        
        # 筛选：1. 概率>=阈值 2. 不与已有线路重叠
        new_predictions = []
        overlap_count = 0
        
        for (city_a, city_b), prob in zip(candidates, probas):
            if prob >= prob_threshold:
                is_overlap = check_overlap(city_a, city_b, cumulative_routes, cities_dict)
                if not is_overlap:
                    new_predictions.append((city_a, city_b, prob))
                else:
                    overlap_count += 1
        
        new_predictions.sort(key=lambda x: x[2], reverse=True)
        
        year_end = time.time()
        
        print(f"    预测新建: {len(new_predictions)}条")
        print(f"    过滤重叠线路: {overlap_count}条")
        print(f"    本预测耗时: {year_end - year_start:.1f}秒")
        
        if new_predictions:
            print("    Top 5 预测线路:")
            for city_a, city_b, prob in new_predictions[:5]:
                print(f"      {city_a} ↔ {city_b}: {prob:.2%}")
        
        # 添加新预测到累积线路
        for city_a, city_b, prob in new_predictions:
            cumulative_routes.append((city_a, city_b))
        
        all_predictions[year] = new_predictions
        sys.stdout.flush()
    
    total_time = time.time() - start_time_total
    print(f"\n[统计] 总预测耗时: {total_time:.1f}秒")
    
    return all_predictions


def time_split_validation(cities_df, history_df, train_end_year=2020, val_start_year=2021, val_end_year=2025):
    """
    时间分割验证：用历史数据预测未来，计算真实准确率
    
    Args:
        cities_df: 城市数据
        history_df: 历史线路数据
        train_end_year: 训练集截止年份
        val_start_year: 验证集开始年份
        val_end_year: 验证集结束年份
    
    Returns:
        dict: 验证指标
    """
    print("\n" + "="*60)
    print("【时间分割验证】")
    print(f"训练集: 2008-{train_end_year}")
    print(f"验证集: {val_start_year}-{val_end_year}")
    print("="*60)
    
    cities_dict = {row['name']: row for _, row in cities_df.iterrows()}
    city_names = set(cities_df['name'])
    
    # 构建训练集（2008-2020）
    train_pairs = set()
    for idx, row in history_df.iterrows():
        year = int(row['open_year'])
        if year <= train_end_year:
            start = str(row['start_city']).strip()
            end = str(row['end_city']).strip()
            if start in city_names and end in city_names:
                normalized = tuple(sorted([start, end]))
                train_pairs.add(normalized)
    
    # 构建验证集（2021-2025，并且不在训练集中的）
    val_true_pairs = set()
    for idx, row in history_df.iterrows():
        year = int(row['open_year'])
        if val_start_year <= year <= val_end_year:
            start = str(row['start_city']).strip()
            end = str(row['end_city']).strip()
            if start in city_names and end in city_names:
                normalized = tuple(sorted([start, end]))
                if normalized not in train_pairs:  # 只取训练后新建的线路
                    val_true_pairs.add(normalized)
    
    print(f"\n训练集线路数: {len(train_pairs)}")
    print(f"验证集真实线路数: {len(val_true_pairs)}")
    
    # 训练数据构建
    X_train = []
    y_train = []
    
    # 正样本：训练集线路
    for city_a, city_b in train_pairs:
        feat = generate_enhanced_features(cities_df, cities_dict, city_a, city_b, train_pairs)
        if feat:
            X_train.append(feat)
            y_train.append(1)
    
    # 负样本：随机未连接的
    all_city_list = list(city_names)
    np.random.seed(42)
    neg_count = 0
    attempts = 0
    while neg_count < len(train_pairs) * 3 and attempts < 10000:
        attempts += 1
        city_a = np.random.choice(all_city_list)
        city_b = np.random.choice(all_city_list)
        if city_a == city_b:
            continue
        pair = tuple(sorted([city_a, city_b]))
        if pair not in train_pairs and pair not in val_true_pairs:
            feat = generate_enhanced_features(cities_df, cities_dict, city_a, city_b, train_pairs)
            if feat:
                X_train.append(feat)
                y_train.append(0)
                neg_count += 1
    
    # 训练模型
    print(f"\n训练样本数: {len(X_train)} (正: {len(train_pairs)}, 负: {neg_count})")
    
    X_df = pd.DataFrame(X_train)
    y_array = np.array(y_train)
    
    if XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.15,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_df, y_array)
    else:
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_df, y_array)
    
    # 验证集预测：所有在2021-2025新建的线路 + 随机负样本
    X_val = []
    y_val = []
    
    # 验证正样本
    for city_a, city_b in val_true_pairs:
        feat = generate_enhanced_features(cities_df, cities_dict, city_a, city_b, train_pairs)
        if feat:
            X_val.append(feat)
            y_val.append(1)
    
    # 验证负样本
    val_neg_count = 0
    attempts = 0
    while val_neg_count < len(val_true_pairs) * 3 and attempts < 10000:
        attempts += 1
        city_a = np.random.choice(all_city_list)
        city_b = np.random.choice(all_city_list)
        if city_a == city_b:
            continue
        pair = tuple(sorted([city_a, city_b]))
        if pair not in train_pairs and pair not in val_true_pairs:
            feat = generate_enhanced_features(cities_df, cities_dict, city_a, city_b, train_pairs)
            if feat:
                X_val.append(feat)
                y_val.append(0)
                val_neg_count += 1
    
    print(f"验证样本数: {len(X_val)} (正: {len(val_true_pairs)}, 负: {val_neg_count})")
    
    # 预测验证集
    X_val_df = pd.DataFrame(X_val)
    y_pred_proba = model.predict_proba(X_val_df)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # 计算指标
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    print(f"\n【验证结果】")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"准确率: {acc:.4f}")
    print(f"分类报告:")
    print(classification_report(y_val, y_pred))
    
    # 保存验证结果
    validation_report = {
        "train_end_year": train_end_year,
        "val_start_year": val_start_year,
        "val_end_year": val_end_year,
        "train_pairs": len(train_pairs),
        "val_true_pairs": len(val_true_pairs),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "auc_roc": float(auc),
        "accuracy": float(acc),
        "precision_1": float(report['1']['precision']),
        "recall_1": float(report['1']['recall']),
        "f1_1": float(report['1']['f1-score']),
        "classification_report": report
    }
    
    # 保存到JSON
    import json
    with open("validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=4)
    print("\n[保存] 验证结果 -> validation_results.json")
    
    return validation_report


def save_predictions_to_csv(all_predictions, output_path="predicted_routes_rolling.csv"):
    """保存预测结果"""
    rows = []
    for year, predictions in all_predictions.items():
        for city_a, city_b, prob in predictions:
            rows.append({
                'year': year,
                'city_a': city_a,
                'city_b': city_b,
                'probability': prob
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[保存] 预测结果 -> {output_path}")
    print(f"  总预测线路: {len(df)} 条")
    return df


def main():
    print("="*60)
    print("🚄 中国高铁滚动预测系统 - 增强版")
    print("="*60)
    print("\n【增强特性】")
    print("1. 时间范围: 2026-2030年（优化速度）")
    print("2. 概率阈值: 90%以上")
    print("3. 空间拓扑修正: 过滤夹角<30度的重复线路")
    print("4. 增强特征: 经济/行政/地理/网络/跨海等22个特征")
    print("5. 性能优化: 精简模型参数，提升预测速度")
    
    cities_df = build_city_dataframe(CITIES_DATA)
    history_df = load_history_data()
    
    print(f"\n城市数: {len(cities_df)}")
    print(f"历史线路记录: {len(history_df)}")
    
    # 【步骤1】时间分割验证（真实准确率计算）
    val_results = time_split_validation(
        cities_df, 
        history_df, 
        train_end_year=2020, 
        val_start_year=2021, 
        val_end_year=2025
    )
    
    # 【步骤2】滚动预测
    all_predictions = rolling_prediction(
        cities_df, 
        history_df, 
        start_year=2026, 
        end_year=2030, 
        prob_threshold=0.90
    )
    
    result_df = save_predictions_to_csv(all_predictions)
    
    # 【步骤3】生成完整报告数据
    report_data = {
        "validation": val_results,
        "predictions_summary": {},
        "feature_names": list(generate_enhanced_features(
            cities_df, 
            {row['name']: row for _, row in cities_df.iterrows()},
            cities_df['name'].iloc[0],
            cities_df['name'].iloc[1],
            set()
        ).keys()) if len(cities_df) >= 2 else []
    }
    
    for year, preds in all_predictions.items():
        report_data["predictions_summary"][year] = {
            "count": len(preds),
            "top_5": [{"city_a": c1, "city_b": c2, "probability": float(p)} 
                     for c1, c2, p in preds[:5]]
        }
    
    # 保存完整报告数据
    import json
    with open("report_data.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)
    print("[保存] 报告数据 -> report_data.json")
    
    print("\n" + "="*60)
    print("✅ 滚动预测完成！")
    print("="*60)
    print("\n各年份预测线路数:")
    for year, preds in all_predictions.items():
        print(f"  {year}年: {len(preds)} 条")


if __name__ == "__main__":
    main()
