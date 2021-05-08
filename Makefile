PY=python3

.PHONY: all pyquest clean build install rebuild
all: pyquest
build: pyquest
rebuild:
	$(MAKE) clean
	$(MAKE) pyquest

pyquest:
	$(PY) setup.py build

install:
	$(PY) setup.py install

clean:
	rm -rf _skbuild
