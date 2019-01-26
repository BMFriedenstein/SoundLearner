SUBDIRS := instrument fft midi genetic_trainer player dataset_builder wave

all: src/$(SUBDIRS)
clean: src/$(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C src/$@ $(MAKECMDGOALS)

.PHONY: all $(SUBDIRS) clean