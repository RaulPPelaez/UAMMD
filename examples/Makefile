DIRECTORIES := $(sort $(dir $(wildcard */)))

all: $(DIRECTORIES)


clean: $(foreach dir,$(DIRECTORIES), clean_$(dir))

define makerule
.PHONY: $(1)
$(1):
	+$(MAKE) -C $(1)

clean_$(1):
	$(MAKE) -C $(1) clean
endef

$(foreach dir,$(DIRECTORIES),$(eval $(call makerule,$(dir))))
