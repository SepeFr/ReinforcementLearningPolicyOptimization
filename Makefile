CXX ?= c++

TARGET := main
BUILD_DIR := build/objects

DEMO_SOURCES := main.cpp \
	exps/environments/Pendulum.cpp \
	exps/environments/PendulumStochastic.cpp
LIBRARY_SOURCES := $(shell find src -type f -name '*.cpp' | sort)
SOURCES := $(DEMO_SOURCES) $(LIBRARY_SOURCES)
OBJECTS := $(addprefix $(BUILD_DIR)/,$(SOURCES:.cpp=.o))
DEPENDENCIES := $(OBJECTS:.o=.d)

INCLUDE_DIRS := $(shell find include -type d | sort)
CPPFLAGS += $(addprefix -I,$(INCLUDE_DIRS))
CXXFLAGS ?= -O2
PROJECT_FLAGS := -std=c++20 -Wall -Wextra -Wpedantic
DEPENDENCY_FLAGS := -MMD -MP

.DEFAULT_GOAL := all
.PHONY: all run clean

all: $(TARGET)

run: $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.cpp Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(PROJECT_FLAGS) $(DEPENDENCY_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEPENDENCIES)
