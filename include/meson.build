common_sources = files(
  'filereader.cpp',
  'filewriter.cpp',
)

common_lib = static_library(
  'common',
  common_sources,
  include_directories : root_inc,
  dependencies : [],
  build_by_default : false,
)

common_dep = declare_dependency(
  link_with : common_lib,
  include_directories : root_inc,
)
