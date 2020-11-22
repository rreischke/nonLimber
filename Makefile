CC	= g++

SRC_PATH = ./


LDFLAGS = -shared
CCFLAGS = -O3 -pedantic -Wall -fPIC -m64 -Xpreprocessor -fopenmp -std=c++11 -I$(SRC_PATH)

LIBDIR	= ./lib
INCDIR	= ./include

STATIC	= lib_levin.a


classes = \
	Levin_power.cpp
	
objects	= $(classes:.cpp=.o)
headers	= $(classes:.cpp=.h)

$(STATIC): $(objects)
	ar rv $(STATIC) $(objects)

%.o: %.cpp %.h
	$(CC) $(CCFLAGS) -c $<

.PHONY: clean distclean doc
.DEFAULT_GOAL: static

static: $(STATIC)

doc:
	doxygen

clean:
	$(RM) *.o *~
distclean: clean
	$(RM) Makefile $(STATIC)

install: $(STATIC)
	mv $(STATIC) $(LIBDIR); \
	$(RM) -r doc
