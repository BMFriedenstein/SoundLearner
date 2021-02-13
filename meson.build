project('SoundLearner', 'cpp',
        default_options : [
                           'cpp_std=c++2a',
                           'optimization=0',
                           'warning_level=3',
                           'b_lto=true'
                          ]
)
add_project_arguments('-pedantic', language : 'cpp')

# pthread_dep = dependency('threads')
subdir('include/')
subdir('instrument/')
subdir('dataset_builder/')
subdir('genetic_trainer/')
subdir('player/')

