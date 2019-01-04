CXX=g++
CFLAGS = -Wall -O3 -g -std=c++11 -I../ -I./ -I../libs  
SOURCES = main.cpp
OBJ = $(SOURCES:.cpp=.o)
Target = StringSoundsPlayer
LIBS=common
LDLIBS=$(addprefix -L../libs/, $(LIBS)) $(addprefix -l, $(LIBS))
DEPS=
.PHONY: clean libs

$(Target): | libs $(OBJ)
	$(CXX) -o $@ $(OBJ) $(CFLAGS) $(LDLIBS)
	cp $@ $@.debug
	strip $@

libs:
	$(MAKE) -C ../libs $(MAKECMDGOALS)

%.o: %.cpp
	@echo "COMPILING SOURCE $< INTO OBJECT $@"
	$(CXX) -c $< -o $@  $(CFLAGS)

clean: libs
	rm -rf *.o *.d $(Target) *.debug