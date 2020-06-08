project('SoundLearner', 'cpp',
        default_options : [
                           'cpp_std=gnu++17',
                           'optimization=3',
                           'warning_level=3',
                           'b_lto=true'
                          ]
)

# pthread_dep = dependency('threads')
subdir('include/')
subdir('instrument/')
subdir('dataset_builder/')
subdir('genetic_trainer/')
subdir('player/')

