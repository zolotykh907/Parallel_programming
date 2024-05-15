INFO = -Minfo=all
LIBS = -lboost_program_options
GPU = -acc=gpu
CXX = pgc++

all:gpu
	
gpu: main2.cpp
	$(CXX) $(GPU) $(INFO) $(LIBS) -o $@ $<

clean:all
	rm gpu
