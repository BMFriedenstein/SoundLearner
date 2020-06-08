CXX=icc
CFLAGS=-Wall -O2 -xHASWELL -ansi-alias -std=c++14 -I./ -I../ -I../../ -I../../ -I$TACC_FFTW3_INC
SOURCES=fft.cpp img.cpp
OBJ=$(SOURCES:.cpp=.o)
DEPS=$(SOURCES:.cpp=.d)
LDLIBS=-lfftw3
TARGET=libfft.a
-include $(DEPS)

.PHONY: clean
all: $(TARGET)

$(TARGET): $(OBJ)
	ar crf $@ $(OBJ)
	cp $@ ../../shared/

%.o: %.cpp
	$(CXX) -c $< -o $@  $(CFLAGS) $(LDLIBS)

clean:
	rm -rf *.o *.d $(TARGET) ../../shared/$(TARGET)