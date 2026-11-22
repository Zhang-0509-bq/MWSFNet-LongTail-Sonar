# config.py：统一管理数据集配置
DATASET_LIST = [
                "FLSMDD",
                "NKSID"
                ]
DATASET_CLASSES = {
    'FLSMDD': 10,
    'NKSID': 8
}
OLD_NEW_LBL_MAP_DICT = {
    "FLSMDD": {cls: i for i, cls in enumerate(range(10))},
    "NKSID": {cls: i for i, cls in enumerate(range(8))}
}
DATA_ROOT = "../data"
