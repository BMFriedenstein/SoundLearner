incdir = include_directories('../')
libinstrument = static_library( 'instrument',
                                'instrument_model.cpp',
                                'string_oscillator.cpp',
                                 dependencies : [ common_dep ],
                                 build_by_default: false
                          )

instrument_dep = declare_dependency( link_with : libinstrument,
                                     include_directories : incdir, )