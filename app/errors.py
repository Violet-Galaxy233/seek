class SeekError(Exception):
    """Seek 可向调用者说明的基础错误。"""


class ModelUnavailableError(SeekError):
    """本地模型不存在或不能加载。"""


class CatalogError(SeekError):
    """测试数据目录或目录清单无效。"""


class IndexError(SeekError):
    """FAISS 索引不能建立或读取。"""


class RemoteDataError(SeekError):
    """外部元数据或封面服务请求失败。"""


class ImportDataError(SeekError):
    """远端记录无法转换为本地专辑数据。"""
