import os
import io
import pickle
import gzip
import lzma
import bz2

# https://stackoverflow.com/questions/12332975/installing-python-module-within-code
def install(package):
    import pip
    if hasattr(pip, "main"):
        pip.main(["install", package])
    else:
        import pip._internal
        pip._internal.main(["install", package])

def _read(path, compression, mode="rb", pickling=True):
    with compression.open(path, mode=mode) as f:
        obj = f.read()
    if pickling:
        obj = pickle.loads(obj)
    return obj
        
def read(path, mode="rb", pickling=True):
    _, extension = os.path.splitext(path)
    ext_dict = {
        '.gz' : gzip,
        '.gzip' : gzip,
        '.zip' : gzip,
        '.xz' : lzma,
        '.lzma' : lzma,
        '.bz2': bz2,
        '.pkl': io,
        '.pickle' : io,
    }
    if extension in ('.pkl', '.pickle'):
        mode = "rb"
        pickling = True
    compression = ext_dict.get(extension, io)
    return _read(path, compression, mode, pickling)

def _save(path, obj, compression, mode="wb", pickling=True, **kwargs):
    if pickling:
        obj = pickle.dumps(obj)
    with compression.open(path, mode=mode, **kwargs) as f:
        f.write(obj)

def save(path, obj, mode="wb", pickling=True, **kwargs):
    _, extension = os.path.splitext(path)
    ext_dict = {
        '.gz' : gzip,
        '.gzip' : gzip,
        '.zip' : gzip,
        '.xz' : lzma,
        '.lzma' : lzma,
        '.bz2': bz2,
        '.pkl': io,
        '.pickle' : io,
    }
    if extension in ('.pkl', '.pickle'):
        mode = "wb"
        pickling = True
    compression = ext_dict.get(extension, io)
    _save(path, obj, compression, mode, pickling, **kwargs)

        
def StanModel(file=None, charset='utf-8', model_name='anon_model', model_code=None, stanc_ret=None, include_paths=None, boost_lib=None, eigen_lib=None, verbose=False, obfuscate_model_name=True, extra_compile_args=None, cache_path=None, **kwargs):
    """ Stan Colab utils - StanModel
    
    cache_path : string
        Path to read/save the StanModel object.
        If None, don't save
    
    Init signature: pystan.StanModel(file=None, charset='utf-8', model_name='anon_model', model_code=None, stanc_ret=None, include_paths=None, boost_lib=None, eigen_lib=None, verbose=False, obfuscate_model_name=True, extra_compile_args=None)
    Docstring:     
    Model described in Stan's modeling language compiled from C++ code.

    Instances of StanModel are typically created indirectly by the functions
    `stan` and `stanc`.

    Parameters
    ----------
    file : string {'filename', 'file'}
        If filename, the string passed as an argument is expected to
        be a filename containing the Stan model specification.

        If file, the object passed must have a 'read' method (file-like
        object) that is called to fetch the Stan model specification.

    charset : string, 'utf-8' by default
        If bytes or files are provided, this charset is used to decode.

    model_name: string, 'anon_model' by default
        A string naming the model. If none is provided 'anon_model' is
        the default. However, if `file` is a filename, then the filename
        will be used to provide a name.

    model_code : string
        A string containing the Stan model specification. Alternatively,
        the model may be provided with the parameter `file`.

    stanc_ret : dict
        A dict returned from a previous call to `stanc` which can be
        used to specify the model instead of using the parameter `file` or
        `model_code`.

    include_paths : list of strings
        Paths for #include files defined in Stan program code.

    boost_lib : string
        The path to a version of the Boost C++ library to use instead of
        the one supplied with PyStan.

    eigen_lib : string
        The path to a version of the Eigen C++ library to use instead of
        the one in the supplied with PyStan.

    verbose : boolean, False by default
        Indicates whether intermediate output should be piped to the console.
        This output may be useful for debugging.

    kwargs : keyword arguments
        Additional arguments passed to `stanc`.

    Attributes
    ----------
    model_name : string
    model_code : string
        Stan code for the model.
    model_cpp : string
        C++ code for the model.
    module : builtins.module
        Python module created by compiling the C++ code for the model.

    Methods
    -------
    show
        Print the Stan model specification.
    sampling
        Draw samples from the model.
    optimizing
        Obtain a point estimate by maximizing the log-posterior.
    get_cppcode
        Return the C++ code for the module.
    get_cxxflags
        Return the 'CXXFLAGS' used for compiling the model.
    get_include_paths
        Return include_paths used for compiled model.

    See also
    --------
    stanc: Compile a Stan model specification
    stan: Fit a model using Stan

    Notes
    -----

    More details of Stan, including the full user's guide and
    reference manual can be found at <URL: http://mc-stan.org/>.

    There are three ways to specify the model's code for `stan_model`.

    1. parameter `model_code`, containing a string to whose value is
       the Stan model specification,

    2. parameter `file`, indicating a file (or a connection) from
       which to read the Stan model specification, or

    3. parameter `stanc_ret`, indicating the re-use of a model
         generated in a previous call to `stanc`.

    References
    ----------

    The Stan Development Team (2013) *Stan Modeling Language User's
    Guide and Reference Manual*.  <URL: http://mc-stan.org/>.

    Examples
    --------
    >>> model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
    >>> model_code; m = StanModel(model_code=model_code)
    ... # doctest: +ELLIPSIS
    'parameters ...
    >>> m.model_name
    'anon_model'
    """
    try:
        import pystan
    except:
        install("pystan")
    
    if cache_path is not None and os.path.exists(cache_path):
        try:
            stan_model = read(cache_path)
            compile_model = False
        except (OSError, TypeError):
            compile_model = True
    else:
         compile_model = True
    
    if compile_model:
        stan_model_kwargs = dict(
            file=file, 
            charset=charset, 
            model_name=model_name, 
            model_code=model_code, 
            stanc_ret=stanc_ret, 
            include_paths=include_paths, 
            boost_lib=boost_lib, 
            eigen_lib=eigen_lib, 
            verbose=verbose, 
            obfuscate_model_name=obfuscate_model_name, 
            extra_compile_args=extra_compile_args,  
            **kwargs
        )
        stan_model = pystan.StanModel(**stan_model_kwargs)
        if cache_path:
            save(cache_path, stan_model)
    
    return stan_model