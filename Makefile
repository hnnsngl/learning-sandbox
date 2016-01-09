ifeq ($(CXX), icc)
OPENMP+=-qopenmp
else
OPENMP+=-fopenmp
endif

CXXFLAGS=-std=c++11 -O3 $(OPENMP) -DNDEBUG $$(pkg-config --cflags opencv)
CXXFLAGS_DEBUG=-std=c++11 $(OPENMP) -g -DDEBUG $$(pkg-config --cflags opencv)

LDFLAGS=$$(pkg-config --libs opencv) -L/usr/lib/x86_64-linux-gnu/ -lboost_program_options

BINARIES=mnist-static mnist-flex
BINARIES_DEBUG=$(BINARIES:=-debug)

CPP_FILES=$(shell ls *cpp)
DEP_FILES=$(CPP_FILES:.cpp=.d)
OBJ_FILES=$(CPP_FILES:.cpp=.o)

.PHONY: all clean help

all: $(BINARIES) $(BINARIES_DEBUG)

clean:
	@rm -f $(BINARIES)
	@rm -f $(BINARIES_DEBUG)
	@rm -f $(OBJ_FILES)
	@rm -f $(DEP_FILES)

help:
	@printf "Binaries\t%s\n" $(BINARIES)
	@printf "Sources \t%s\n" $(CPP_FILES)
	@printf "Object  \t%s\n" $(OBJ_FILES)

include $(DEP_FILES)

# build targets

%-debug: CXXFLAGS=$(CXXFLAGS_DEBUG)

mnist-static: mnist-static.o loadweights.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

mnist-static-debug: mnist-static-debug.o loadweights.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

mnist-flex: mnist-flex.o loadweights.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

mnist-flex-debug: mnist-flex-debug.o loadweights.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

# implicit rules

%.d: %.cpp
	$(CXX) -MM $(CXXFLAGS) $*.cpp > $*.d

%.o: %.cpp %.d
	$(CXX) -c $(CXXFLAGS) -o $@ $< $(LDFLAGS)

%-debug.o: %.cpp %.d
	$(CXX) -c $(CXXFLAGS_DEBUG) -o $@ $< $(LDFLAGS)





