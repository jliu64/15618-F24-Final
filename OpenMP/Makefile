# Application name
APP_NAME = flightroute

# Object files
OBJS = flightroute.o

# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -O3 -std=c++17 -m64 -I. -fopenmp -Wno-unknown-pragmas

# Build the application
all: $(APP_NAME)

# Linking rule
$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Compilation rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class
