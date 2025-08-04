from pickle import dump, load


class SerializerMixin:
    def save(self, path: str):
        with open(path, "wb") as f:
            dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return load(f)
