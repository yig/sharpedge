class DebugOptions:
    def __init__(self, show_plot=False, save_gltf=False, verbose=False):
        self.show_plot = show_plot
        self.save_gltf = save_gltf
        self.verbose = verbose

    def plot(self, fn, *args, **kwargs):
        if self.show_plot:
            fn(*args, **kwargs)

    def save(self, fn, *args, **kwargs):
        if self.save_gltf:
            fn(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
