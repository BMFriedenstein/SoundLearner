CXX=g++
CFLAGS = -Wall -O3 -std=c++11 -I../ -I./
SOURCES = midi.cpp
OBJ = $(SOURCES:.cpp=.o)
DEPS=
.PHONY: clean
Target=libmidi.a

all: $(Target)

$(Target): $(OBJ)
	ar crf $@ $(OBJ)
	cp $@ ../include/

%.o: %.cpp
	@echo "COMPILING SOURCE $< INTO OBJECT $@"
	$(CXX) -c $< -o $@  $(CFLAGS)

clean:
	rm -rf *.o *.d $(Target)