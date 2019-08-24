CXX=icc
CFLAGS=-Wall -02 -xHASWELL -ansi-alias -g -std=c++14 -I./ -I../ -I../../
SOURCES=main.cpp
OBJ=$(SOURCES:.cpp=.o)
TARGET = StringSoundsPlayer
LIBS= midi wave instrument
LDLIBS= -L../../shared $(addprefix -l, $(LIBS))
DEPS=
.PHONY: clean

$(TARGET): $(OBJ) $(LIBS)
	$(CXX) -o $@ $(OBJ) $(CFLAGS) $(LDLIBS)
	cp $@ $@.debug
	strip $@

$(LIBS):
	$(MAKE) -C ../$@ $(MAKECMDGOALS)

%.o: %.cpp
	$(CXX) -c $< -o $@  $(CFLAGS)

clean: $(LIBS)	
	rm -rf *.o *.d $(TARGET) *.debug