SUBDIRS := instrument midi trainer player 

all: src/$(SUBDIRS)
clean: src/$(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C src/$@ $(MAKECMDGOALS)

.PHONY: all $(SUBDIRS) clean