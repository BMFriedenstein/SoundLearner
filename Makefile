SUBDIRS :=libs trainer player 

all: $(SUBDIRS)
clean: $(SUBDIRS)
make_trainer: trainer
make_player: player
make_libs: libs

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: all $(SUBDIRS) clean