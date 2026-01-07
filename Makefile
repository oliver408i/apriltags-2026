PYTHON ?= python3
APRILTAG_BUILD_DIR ?= build/apriltag
AT_STATIC_LIB = $(APRILTAG_BUILD_DIR)/libapriltag.a

.PHONY: all apriltag cython clean help

all: apriltag cython

apriltag: $(AT_STATIC_LIB)

$(AT_STATIC_LIB):
	cmake -S src/apriltag_lib -B $(APRILTAG_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DCMAKE_C_FLAGS="-march=native"
	cmake --build $(APRILTAG_BUILD_DIR) --config Release

cython: $(AT_STATIC_LIB)
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf $(APRILTAG_BUILD_DIR)
	rm -f vision_engine*.so
	rm -f src/*.c  # Clean up the intermediate Cython C files

help:
	@echo "make          - Build everything"
	@echo "make clean    - Remove build artifacts"
	@echo "make apriltag - Only build the C library"