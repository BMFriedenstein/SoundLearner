CXX=g++
CFLAGS=-Wall -O3 -g -std=c++11 -I./ -I../ -I../../ 
SOURCES=main.cpp
OBJ=$(SOURCES:.cpp=.o)
TARGET = StringSoundsPlayer
LIBS= midi wave instrument
LDLIBS= -L../../include $(addprefix -l, $(LIBS))
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

clean:
	rm -rf *.o *.d $(TARGET) *.debug