from pycuda.compiler import DynamicModule
from pynvrtc.interface import NVRTCInterface, NVRTCException

# Dependency: pynvrtc (pip install pynvrtc, https://github.com/NVIDIA/pynvrtc )
class nvrtcSourceModule(DynamicModule):
    # FIXME: Check relevance of each argument
    def __init__(self, source, nvcc="nvcc", options=None, keep=False,
            no_extern_c=False, arch=None, code=None, cache_dir=None,
            include_dirs=None, cuda_libdir=None):
        if include_dirs is None:
            include_dirs = []
        super(nvrtcSourceModule, self).__init__(nvcc=nvcc,
            link_options=None, keep=keep, no_extern_c=no_extern_c,
            arch=arch, code=code, cache_dir=cache_dir,
            include_dirs=include_dirs, cuda_libdir=cuda_libdir)
        if options is None:
            options = DEFAULT_NVCC_FLAGS
        options = options[:]
        #if '-rdc=true' not in options:
        #    options.append('-rdc=true')
        #if '-lcudadevrt' not in options:
        #    options.append('-lcudadevrt')

        self._entries = { options[i+1]: None for i in range(len(options)-1) if options[i] == '-e' }

        self._nvrtc = NVRTCInterface() # lib_path=path-to-libnvrtc.so ?
        self._prog = None

        self.add_source(source, nvcc_options=options)
        self.add_stdlib('cudadevrt')
        self.link()
    def __del__(self):
        if self._prog is not None:
            self._nvrtc.nvrtcDestroyProgram(self._prog)

    def add_source(self, source, nvcc_options=None, name='kernel.ptx'):
        if nvcc_options is None:
            nvcc_options = []

        # TODO: Does the program name matter? Use 'name' argument maybe?
        self._prog = self._nvrtc.nvrtcCreateProgram(source, "default_program", [], [])
        # let kernels be generated for the requested entries
        for name in self._entries.keys():
            self._nvrtc.nvrtcAddNameExpression(self._prog, name)
        try:
            # TODO: figure out options
            self._nvrtc.nvrtcCompileProgram(self._prog, options=[]) # nvcc_options
        except NVRTCException as e:
            # TODO: Report errors more cleanly
            print(self._nvrtc.nvrtcGetProgramLog(self._prog))
            raise

        # get the mangled ("lowered") names for the requested entries.
        # pycuda needs these to launch the kernels.
        for name in self._entries.keys():
            self._entries[name] = self._nvrtc.nvrtcGetLoweredName(self._prog, name)

        ptx = self._nvrtc.nvrtcGetPTX(self._prog).encode("utf-8")
        from pycuda.driver import jit_input_type
        self.linker.add_data(ptx, jit_input_type.PTX, name)
        return self
    def get_function(self, name):
        try:
            name = self._entries[name]
        except KeyError:
            pass

        return super(nvrtcSourceModule, self).get_function(name)


