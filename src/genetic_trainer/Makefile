CXX=g++
CFLAGS=-Wall -O3 -g -std=c++14 -I./ -I../ -I../../
SOURCES=main.cpp trainer.cpp 
OBJS=$(SOURCES:.cpp=.o)
TARGET= StringSoundsTrainer
LIBS= midi wave instrument fft logger
LDLIBS=-L../../include $(addprefix -l, $(LIBS))
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
