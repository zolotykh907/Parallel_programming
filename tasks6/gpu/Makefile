INFO = -Minfo=all
LIBS = -lboost_program_options
GPU = -acc=gpu
CXX = pgc++

all:gpu
	
gpu: main1.cpp
	$(CXX) $(GPU) $(INFO) $(LIBS) -o $@ $<

clean:all
	rm gpu
