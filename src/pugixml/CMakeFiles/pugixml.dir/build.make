# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\msys64\mingw64\bin\cmake.exe

# The command to remove a file.
RM = C:\msys64\mingw64\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\corey\Documents\TransportProject\Branson\branson\src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\corey\Documents\TransportProject\Branson\branson\src

# Include any dependencies generated for this target.
include pugixml/CMakeFiles/pugixml.dir/depend.make

# Include the progress variables for this target.
include pugixml/CMakeFiles/pugixml.dir/progress.make

# Include the compile flags for this target's objects.
include pugixml/CMakeFiles/pugixml.dir/flags.make

pugixml/CMakeFiles/pugixml.dir/src/pugixml.cpp.obj: pugixml/CMakeFiles/pugixml.dir/flags.make
pugixml/CMakeFiles/pugixml.dir/src/pugixml.cpp.obj: pugixml/CMakeFiles/pugixml.dir/includes_CXX.rsp
pugixml/CMakeFiles/pugixml.dir/src/pugixml.cpp.obj: pugixml/src/pugixml.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\corey\Documents\TransportProject\Branson\branson\src\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object pugixml/CMakeFiles/pugixml.dir/src/pugixml.cpp.obj"
	cd /d C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml && C:\msys64\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\pugixml.dir\src\pugixml.cpp.obj -c C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml\src\pugixml.cpp

pugixml/CMakeFiles/pugixml.dir/src/pugixml.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pugixml.dir/src/pugixml.cpp.i"
	cd /d C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml && C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml\src\pugixml.cpp > CMakeFiles\pugixml.dir\src\pugixml.cpp.i

pugixml/CMakeFiles/pugixml.dir/src/pugixml.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pugixml.dir/src/pugixml.cpp.s"
	cd /d C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml && C:\msys64\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml\src\pugixml.cpp -o CMakeFiles\pugixml.dir\src\pugixml.cpp.s

# Object files for target pugixml
pugixml_OBJECTS = \
"CMakeFiles/pugixml.dir/src/pugixml.cpp.obj"

# External object files for target pugixml
pugixml_EXTERNAL_OBJECTS =

pugixml/libpugixml.a: pugixml/CMakeFiles/pugixml.dir/src/pugixml.cpp.obj
pugixml/libpugixml.a: pugixml/CMakeFiles/pugixml.dir/build.make
pugixml/libpugixml.a: pugixml/CMakeFiles/pugixml.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\corey\Documents\TransportProject\Branson\branson\src\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libpugixml.a"
	cd /d C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml && $(CMAKE_COMMAND) -P CMakeFiles\pugixml.dir\cmake_clean_target.cmake
	cd /d C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\pugixml.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
pugixml/CMakeFiles/pugixml.dir/build: pugixml/libpugixml.a

.PHONY : pugixml/CMakeFiles/pugixml.dir/build

pugixml/CMakeFiles/pugixml.dir/clean:
	cd /d C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml && $(CMAKE_COMMAND) -P CMakeFiles\pugixml.dir\cmake_clean.cmake
.PHONY : pugixml/CMakeFiles/pugixml.dir/clean

pugixml/CMakeFiles/pugixml.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\corey\Documents\TransportProject\Branson\branson\src C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml C:\Users\corey\Documents\TransportProject\Branson\branson\src C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml C:\Users\corey\Documents\TransportProject\Branson\branson\src\pugixml\CMakeFiles\pugixml.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : pugixml/CMakeFiles/pugixml.dir/depend

