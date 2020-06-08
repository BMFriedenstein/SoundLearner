CXX=icc
CFLAGS=-O2 -Wall -xHASWELL -ansi-alias -std=c++14 -I../ -I./ -I../../ 
SOURCES=string_oscillator.cpp instrument_model.cpp
OBJ=$(SOURCES:.cpp=.o)
TARGET=libinstrument.a
DEPS=$(SOURCES:.cpp=.d)
-include $(DEPS)

.PHONY: clean
all: $(TARGET)

$(TARGET): $(OBJ)
	ar crf $@ $(OBJ)
	cp $@ ../../shared/

%.o: %.cpp
	$(CXX) -c $< -o $@  $(CFLAGS)

clean:
	rm -rf *.o *.d $(Target) ../../include/$(TARGET)
	