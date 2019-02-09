SUBDIRS := instrument fft midi genetic_trainer player dataset_builder wave
all: $(SUBDIRS)
clean: $(SUBDIRS)

include/.:

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: all $(SUBDIRS) clean