# Makefile

.PHONY: all single threaded clean

.DEFAULT: all

install: python
	pip install -e ./

interface:
	cd sagym/interface; swig -c++ -python -py3 sa_interface.i

python: interface
	python setup.py build_ext --inplace

clean:
	-find . -name sa_interface_wrap.cxx | xargs rm
	-find . -name sa_interface.py | xargs rm
	-find . -name '_sa_interface*.so' | xargs rm
	-find . -name '__pycache__' | xargs rm -r
	-rm -r sagym/interface/build
	-rm -r *.egg-info build/
