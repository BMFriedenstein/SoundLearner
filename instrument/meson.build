instrument_sources = files(
  'instrument_model.cpp',
  'string_oscillator.cpp',
)

libinstrument = static_library(
  'instrument',
  instrument_sources,
  dependencies : common_dep,
  build_by_default : false,
)

instrument_dep = declare_dependency(
  link_with : libinstrument,
  dependencies : common_dep,
)
