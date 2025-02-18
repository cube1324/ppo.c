CC = gcc
NVCC = nvcc
CFLAGS = -Ofast
NVCCFLAGS = -O3
LDFLAGS = -lm -lpython3.10 -lopenblas

SRCDIR = src
CUDASRCDIR = cudasrc
OBJDIR = obj
BINDIR = bin
HEADERS = include
HEADERS_PYTHON = $(PPO_PYTHON)/include/python3.10

LINK_PYTHON = $(PPO_PYTHON)/lib

SOURCES = $(wildcard $(SRCDIR)/*.c)
CU_SOURCES = $(wildcard $(CUDASRCDIR)/*.cu)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
CU_OBJECTS = $(CU_SOURCES:$(CUDASRCDIR)/%.cu=$(OBJDIR)/%.o)
EXECUTABLE = $(BINDIR)/ppo
CUDA_EXECUTABLE = $(BINDIR)/ppo_cuda

all: ppo ppo_cuda

ppo: $(EXECUTABLE)

ppo_cuda: $(CUDA_EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -I $(HEADERS) -I $(HEADERS_PYTHON) -L $(LINK_PYTHON) $(OBJECTS) -o $@ $(LDFLAGS)

$(CUDA_EXECUTABLE): $(CU_OBJECTS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -I $(HEADERS) -I $(HEADERS_PYTHON) -L $(LINK_PYTHON) $(CU_OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) -I $(HEADERS) -I $(HEADERS_PYTHON) -L $(LINK_PYTHON) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(CUDASRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) -I $(HEADERS) -I $(HEADERS_PYTHON) -L $(LINK_PYTHON) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean ppo ppo_cuda