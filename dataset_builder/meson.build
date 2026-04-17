dataset_builder_sources = files(
  'dataset_builder.cpp',
)

fftw_dep = dependency('fftw3', required : false)
if not fftw_dep.found()
  fftw_dep = cpp.find_library('fftw3', required : false)
endif
if not fftw_dep.found()
  fftw_dep = cpp.find_library('libfftw3')
endif

executable(
  'dataset_builder',
  dataset_builder_sources,
  dependencies : [
    instrument_dep,
    fftw_dep,
  ],
  install : false,
)
