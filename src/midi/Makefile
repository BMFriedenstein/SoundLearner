CXX=g++
CFLAGS=-Wall -O3 -std=c++11 -I./ -I../ -I../../
SOURCES=midi.cpp
OBJ=$(SOURCES:.cpp=.o)
DEPS=$(SOURCES:.cpp=.d)
TARGET=libmidi.a

-include $(DEPS)
.PHONY: clean

all: $(TARGET)

$(TARGET): $(OBJ)
	ar crf $@ $(OBJ)
	cp $@ ../../include/

%.o: %.cpp
	$(CXX) -c $< -o $@  $(CFLAGS)

clean:
	rm -rf *.o *.d $(TARGET)