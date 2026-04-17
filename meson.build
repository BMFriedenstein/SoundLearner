project(
  'SoundLearner',
  'cpp',
  version : '0.1.0',
  license : 'MIT',
  meson_version : '>=1.2.0',
  default_options : [
    'cpp_std=c++23',
    'warning_level=3',
    'werror=false',
  ],
)

cpp = meson.get_compiler('cpp')
root_inc = include_directories('.')

fftw_dep = dependency('fftw3', required : false)
if not fftw_dep.found()
  fftw_dep = cpp.find_library('fftw3', required : false)
endif
if not fftw_dep.found()
  fftw_dep = cpp.find_library('libfftw3')
endif

add_project_arguments(
  cpp.get_supported_arguments('-pedantic'),
  language : 'cpp',
)

subdir('include')
subdir('instrument')
subdir('dataset_builder')
subdir('feature_extractor')
subdir('player')
