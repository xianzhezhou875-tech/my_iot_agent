import logging

from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class SensorBuffer:
    def __init__(self):
        # 列表：存储所有记录（时间序）
        self.log_list = []
        # 字典：存储最新的值（键值映射）
        self.id_map = {}

    def add_record(self, sensor_id, value):
        # 1. 存储到历史记录列表
        record = {"id": sensor_id, "value": value}
        self.log_list.append(record)
        
        # 2. 存储到哈希表（字典）中，利用 Key 映射，如果存在则覆盖更新
        self.id_map[sensor_id] = value
        logger.info("记录已存入: %s -> %s", sensor_id, value)

    def get_latest(self, sensor_id):
        # 利用字典的哈希查找特性，不需要遍历，直接通过 key 取值
        # 如果 id 不存在，返回 None
        return self.id_map.get(sensor_id, "未找到该传感器数据")

    def get_history(self):
        # 返回整个列表
        return self.log_list

# --- 测试代码 ---
if __name__ == "__main__":
    buffer = SensorBuffer()

    buffer.add_record("Temp_01", 25.5)
    buffer.add_record("Humid_01", 60.2)
    buffer.add_record("Temp_01", 26.1)

    logger.info("Temp_01 最新温度: %s", buffer.get_latest("Temp_01"))
    logger.info("所有历史记录: %s", buffer.get_history())