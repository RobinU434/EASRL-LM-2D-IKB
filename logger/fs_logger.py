import time 
from pandas import DataFrame

class FileSystemLogger:
    def __init__(self, path) -> None:
        self._path = path

        self._data = []

    def add_scalar(self, entity, y, *argv):
        x_values = {}
        for idx, arg in enumerate(argv):
            x_values[f"x{idx}"] = arg
        
        self._data.append({
            "entity": entity,
            "y": y,
            **x_values,
            "time": time.time()
        })

    def dump(self, file_name: str = "results.csv"):
        df = DataFrame(self._data)
        df.to_csv(self._path + "/" + file_name)

    @property
    def path(self):
        return self._path


if __name__ == "__main__":
    logger = FileSystemLogger(".")

    logger.add_scalar("a", 1, 1)
    logger.add_scalar("a", 2, 2)
    logger.add_scalar("a", 3, 3)
    
    logger.add_scalar("b", 4, 4)
    logger.add_scalar("b", 5, 5)
    logger.add_scalar("b", 6, 6)

    logger.add_scalar("c", 4, 4, 1)
    logger.add_scalar("c", 5, 5, 2)
    logger.add_scalar("c", 6, 6, 3)

    logger.add_scalar("d", [4, 1], 4, 1)
    logger.add_scalar("d", [5, 1], 5, 2)
    logger.add_scalar("d", [6, 1], 6, 3)

    df = DataFrame(logger._data)

    print(df)