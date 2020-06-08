CXX=icc
CFLAGS=-Wall -02 -xHASWELL -ansi-alias -std=c++14 -I./ -I../ -I../../
SOURCES=wave.cpp
OBJ=$(SOURCES:.cpp=.o)
DEPS=$(SOURCES:.cpp=.d)
TARGET=libwave.a
-include $(DEPS)

.PHONY: clean
all: $(TARGET)

$(TARGET): $(OBJ)
	ar crf $@ $(OBJ)
	cp $@ ../../shared/

%.o: %.cpp
	$(CXX) -c $< -o $@  $(CFLAGS)

clean:
	rm -rf *.o *.d $(TARGET) ../../include/$(TARGET)