PROJ=peticodiac

SRCS_DIR=src
OBJS_DIR=obj
HEADER_DIR=include

SRCS=$(wildcard $(SRCS_DIR)/*.cpp)
OBJS=$(patsubst $(SRCS_DIR)/%.cpp,$(OBJS_DIR)/%.o,$(SRCS))
HEADERS=$(wildcard $(HEADER_DIR)/*.h)

# Check for Mac OS since it has a different compiler name
OS=$(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN=$(strip $(findstring DARWIN, $(OS)))

ifneq ($(DARWIN),)
	# MacOS System
	CC=g++-6
else
	# Non MacOS System
	CC=g++
endif

CFLAGS=-Wall -DDEBUG -std=c++11 -I$(HEADER_DIR) -fopenmp
LDFLAGS=-fopenmp

all:$(OBJS)
	$(CC) $(OBJS) -o $(PROJ) $(LDFLAGS)

$(OBJS_DIR)/%.o: $(SRCS_DIR)/%.cpp | $(OBJS_DIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(OBJS_DIR):
	mkdir $(OBJS_DIR)

clean:
	rm -rf $(OBJS_DIR)
	rm -rf $(PROJ)
