SUBDIRS := $(wildcard */.)

all: $(SUBDIRS)
clean: $(SUBDIRS)

include/.:

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: all $(SUBDIRS) clean