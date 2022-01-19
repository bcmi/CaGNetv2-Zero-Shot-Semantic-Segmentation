from .cocostuff import CocoStuff10k, CocoStuff164k, LoaderZLS


def get_dataset(name):
    return {"cocostuff10k": CocoStuff10k, "cocostuff164k": CocoStuff164k, "LoaderZLS": LoaderZLS}[name]
