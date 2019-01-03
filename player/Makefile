CXX=g++
CFLAGS = -Wall -O3 -g -std=c++11
OBJ = main.o
Target = stringSoundsPlayer
DEPS=
.PHONY: clean

$(Target): $(OBJ)
	echo linking...
	$(CXX) -o $@ $^ $(CFLAGS)
	cp $@ $@.debug
	strip $@

%.o: %.cpp $(DEPS)
	echo building $@...
	$(CXX) -c -o $@ $< $(CFLAGS)

clean:
	rm -rf *.o *.d $(Target)