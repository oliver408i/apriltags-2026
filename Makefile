PYTHON ?= python3
APRILTAG_BUILD_DIR ?= build/apriltag
AT_STATIC_LIB = $(APRILTAG_BUILD_DIR)/libapriltag.a
ENTRY ?= main.py
OUT_NAME = APD_2000

.PHONY: all apriltag cython clean help nuitka

all: apriltag cython

shit: all

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

nuitka: cython
	$(PYTHON) -m nuitka --standalone --onefile \
		--follow-imports \
		--include-package-data=cv2 \
		--output-filename=$(OUT_NAME) \
		$(ENTRY)

clean:
	rm -rf $(APRILTAG_BUILD_DIR)
	rm -rf $(ENTRY:.py=.dist)
	rm -rf $(ENTRY:.py=.build)
	rm -rf $(ENTRY:.py=.onefile-dist)
	rm -f vision_engine*.so
	rm -f src/*.c
	rm -f $(OUT_NAME)

help:
	@echo "make          - Build everything (C and Cython)"
	@echo "make nuitka   - Bundle into a standalone binary (use ENTRY=file.py)"
	@echo "make clean    - Remove build artifacts"