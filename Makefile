PYTHON ?= python
APRILTAG_BUILD_DIR ?= build/apriltag

.PHONY: all apriltag cython clean

all: apriltag cython

apriltag:
	cmake -S src/apriltag_lib -B $(APRILTAG_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
	cmake --build $(APRILTAG_BUILD_DIR)

cython:
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf $(APRILTAG_BUILD_DIR)
	rm -f vision_engine.*.so
