from abc import ABC, abstractmethod


class Patcher(ABC):
    @abstractmethod
    def patch(self):
        pass
