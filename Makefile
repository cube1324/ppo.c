CC = nvcc
CFLAGS =  -O3 # -G -g #-O3
LDFLAGS = -lm -lpython3.10 -lopenblas -lcublas

SRCDIR = src
OBJDIR = obj
BINDIR = bin
HEADERS = include
HEADERS_PYTHON = $(PPO_PYTHON)/include/python3.10

LINK_PYTHON = $(PPO_PYTHON)/lib

C_SOURCES = $(wildcard $(SRCDIR)/*.c)
CU_SOURCES = $(wildcard $(SRCDIR)/*.cu)
C_OBJECTS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(C_SOURCES))
CU_OBJECTS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SOURCES))
OBJECTS = $(C_OBJECTS) $(CU_OBJECTS)
EXECUTABLE = $(BINDIR)/ppo

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -I $(HEADERS) -I $(HEADERS_PYTHON) -L $(LINK_PYTHON) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) -I $(HEADERS) -I $(HEADERS_PYTHON) -L $(LINK_PYTHON) -x c $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(CC) -I $(HEADERS) -I $(HEADERS_PYTHON) -L $(LINK_PYTHON) -x cu $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean