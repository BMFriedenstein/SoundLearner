dataset_builder_sources = files(
  'dataset_builder.cpp',
)

executable(
  'dataset_builder',
  dataset_builder_sources,
  dependencies : [
    instrument_dep,
    fftw_dep,
  ],
  install : false,
)
