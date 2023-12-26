import json
from io import BufferedReader


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BufferedReader):
            return obj.name
        return super().default(obj)
