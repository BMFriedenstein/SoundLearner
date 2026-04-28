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

add_project_arguments(
  cpp.get_supported_arguments(
    '-pedantic',
    '-Wconversion',
    '-Wshadow',
  ),
  language : 'cpp',
)

subdir('include')
subdir('instrument')
subdir('dataset_builder')
subdir('player')
