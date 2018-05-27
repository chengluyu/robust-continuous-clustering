This directory contains mex functions that must be _compiled_ before they can
be called from matlab.

Typically the C++ code for function `myfunc` will be in a file called
`myfunc.cpp` and accompanied by a _blank_ file `myfunc.m`, which simply
contains comments explaining call usage of the function (so that `help myfunc`
returns something useful).

## Compiling (mexing)

I haven't invested too much time trying to be sure these functions compile
anywhere. They should. Why not? All compilers are perfectly standard, right?

### Experimental

I've tried to gather the compilation of all these mex functions into a script.
To attempt to compile all mex functions in this directory, issue:


```matlab
compile_gptoolbox_mex
```

I'll try to keep this up to date. We'll see how that goes.

### One-by-one

These mex functions do not depend on each other, so if you just need a certain
function, you can compile it alone directly: 

```
compile_gptoolbox_mex
```

But **don't hit OK**, hit CTRL+C instead. Now the paths are set as needed. Copy
the relevant chunk from `compile_gptoolbox_mex.m` and compile the individual
mex file. For example,

```matlab
mex( ...
  MEXOPTS{:}, MSSE42, STDCPP11, ...
  LIBIGL_BASE{:},EIGEN_INC, ...
  LIBIGL_LIBCGAL, ...
  CGAL_INC,CGAL_LIB{:},CGAL_FLAGS, ...
  BOOST_INC,BOOST_LIB{:}, ...
  'signed_distance_isosurface.cpp');
```

### Dependencies 

Some of these functions depend on:

 - c++11
   - VS2012 or newer (windows users only)
 - Eigen
 - libigl
 - Embree
 - Cork
 - CGAL
   - boost
 - Mac OS X Foundation and AppKit frameworks

### libigl

Libigl is by default a _header only_ library. You do not need to compile it to
use it (though you do need to compile and link to any dependencies, e.g. CGAL).

If you see a mex command _linking_ to libigl libraries (e.g.
`-L/usr/local/igl/libigl -ligl`, etc) you may safely remove these, so long as
you **also remove** any definition of the static library flag:
`-DIGL_STATIC_LIBRARY`.


### OpenMP

OpenMP is _still_ not well supported by clang. I've configured my `mexopts.sh`
to use gcc4.7 (installed via macports) and enabled openmp via the `-fopenmp`
flag. Some of these mex files will utilize this if enabled.

### Windows

Currently, none of these mex functions are developed or even known to be
compilable on Windows. If you succeed, drop me a line. If you fail, I'll try to
help, but I do not have access to a windows machine.
