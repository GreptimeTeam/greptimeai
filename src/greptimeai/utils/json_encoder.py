import json
from io import BufferedReader

from greptimeai import logger


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BufferedReader):
            if hasattr(obj, "name"):
                return obj.name
            logger.warning("buffer obj has no name attribute")
            return {}
        return super().default(obj)
