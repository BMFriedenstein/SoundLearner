incdir = include_directories('../')


cc = meson.get_compiler('cpp')
libfftw_dep = cc.find_library('libfftw3', dirs: '/usr/local/lib/', required: true)
executable('dataset_builder',
           'dataset_builder.cpp',
            include_directories : incdir,
            dependencies : [ instrument_dep, libfftw_dep, common_dep ],
            install : false,
          )