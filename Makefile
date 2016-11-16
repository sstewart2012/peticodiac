SRCS_DIR=src
OBJS_DIR=obj
HEADER_DIR=include

SRCS=$(wildcard $(SRCS_DIR)/*.cpp)
OBJS=$(patsubst $(SRCS_DIR)/%.cpp,$(OBJS_DIR)/%.o,$(SRCS))
HEADERS=$(wildcard $(HEADER_DIR)/*.h)

EXECUTABLE=peticodiac

CC=g++-6
CFLAGS=-Wall -DDEBUG -std=c++11 -I$(HEADER_DIR) -fopenmp
LDFLAGS=-fopenmp

all:$(OBJS)
	$(CC) $(OBJS) -o $(EXECUTABLE) $(LDFLAGS)

$(OBJS_DIR)/%.o: $(SRCS_DIR)/%.cpp | $(OBJS_DIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(OBJS_DIR):
	mkdir $(OBJS_DIR)

clean:
	rm -rf $(OBJS_DIR)
