mkdir -p Analyse
mkdir -p Analyse/cpp-check-html
rm -rf html/*
cppcheck --enable=all --project=build/compile_commands.json --std=c++17 --xml 2> Analyse/cppcheck.xml
cppcheck-htmlreport --title=Sound-Learner --file=Analyse/cppcheck.xml --report-dir=Analyse/cpp-check-html --source-dir=.
cpplint --recursive --linelength=150 --filter=-whitespace . 2> Analyse/cpplint.out
find . -iname *.h -o -iname *.cpp | xargs clang-format -i --sort-includes --verbose
