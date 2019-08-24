CXX=icc
CFLAGS=-Wall -02 -xHASWELL -ansi-alias -g -std=c++14 -I./ -I../ -I../../
SOURCES=main.cpp trainer.cpp 
OBJS=$(SOURCES:.cpp=.o)
TARGET= StringSoundsTrainer
LIBS= midi wave instrument fft logger fftw3
LDLIBS=-L../../shared $(addprefix -l, $(LIBS))
HEADERS=trainer.h
DEPS:=$(HEADERS:.h=.d)
-include $(DEPS)

$(TARGET): $(OBJS) $(LIBS) 
	$(CXX) -o $@ $(OBJS) $(CFLAGS) $(LDLIBS)
	cp $@ $@.debug
	strip $@

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CFLAGS)

$(LIBS): 
	$(MAKE) -C ../$@ $(MAKECMDGOALS)

clean: $(LIBS)
	rm -rf *.o *.d $(TARGET) *.debug
	
.PHONY: clean libs
