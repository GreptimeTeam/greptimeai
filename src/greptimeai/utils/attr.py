from typing import Any, List, Optional

from greptimeai import logger


# get chained attribute
def get_attr(obj: Any, chains: List[str]) -> Optional[Any]:
    if not obj:
        logger.warning(f"obj is None, has no {chains} attribute")
        return None

    if len(chains) == 0:
        return None

    tmp = obj
    for chain in chains:
        try:
            tmp = getattr(tmp, chain)
            if tmp is None:
                logger.warning(f"has no {chain} attribute")
                return None
        except AttributeError as e:
            logger.error(f"failed to get {chains} attribute: {e}")
            return None

    return tmp


def get_optional_attr(objs: List[Any], chains: List[str]) -> Optional[Any]:
    for obj in objs:
        attr = get_attr(obj, chains)
        if attr:
            return attr
    return None
