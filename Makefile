CC = gcc
CFLAGS = -Wall -O3
LDFLAGS = -lm -lpython3.10

SRCDIR = src
OBJDIR = obj
BINDIR = bin
HEADERS = include
HEADERS_PYTHON = $(PPO_PYTHON)/include/python3.10

LINK_PYTHON = $(PPO_PYTHON)/lib

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
EXECUTABLE = $(BINDIR)/ppo

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -I $(HEADERS) -I $(HEADERS_PYTHON) -L $(LINK_PYTHON) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) -I $(HEADERS) -I $(HEADERS_PYTHON) -L $(LINK_PYTHON) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean