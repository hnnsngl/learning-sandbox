CXXFLAGS=-std=c++11 -fopenmp -O3 -DNDEBUG
CXXFLAGS_DEBUG=-std=c++11 -fopenmp -g -DDEBUG

LDFLAGS=$$(pkg-config --cflags --libs opencv)

BINARIES=mnist-static
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

mnist-static: mnist-static.o loadweights.o
	$(CXX) -o $@ $(CXXFLAGS) $^ $(LDFLAGS)

mnist-static-debug: mnist-static-debug.o loadweights.o
	$(CXX) -o $@ $(CXXFLAGS_DEBUG) $^ $(LDFLAGS)

# implicit rules

%.d: %.cpp
	$(CXX) -MM $(CXXFLAGS) $*.cpp > $*.d

%.o: %.cpp %.d
	$(CXX) -c $(CXXFLAGS) -o $@ $<



