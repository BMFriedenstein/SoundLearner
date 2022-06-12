incdir = include_directories('../')


cc = meson.get_compiler('cpp')
libfftw_dep = cc.find_library('libfftw3')

src_files = files('dataset_builder.cpp')
clangtidy = find_program('clang-tidy', required: false)
if clangtidy.found()
    run_target(
        'tidy',
        command: [
            clangtidy,
            '-checks=*',
            '-p', meson.build_root()
        ] + src_files)
endif

executable('dataset_builder',
           'dataset_builder.cpp',
            include_directories : incdir,
            dependencies : [instrument_dep, libfftw_dep, common_dep],
            install : true,
            install_dir : './'
          )