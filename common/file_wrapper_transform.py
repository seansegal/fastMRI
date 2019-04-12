class FileWrapperTransform:
    def __init__(self, data_transform):
        self.transform = data_transform

    def __call__(self, kspace, target, attrs, fname, slice):
        return self.transform(kspace, target, attrs, fname, slice) + (fname, slice)