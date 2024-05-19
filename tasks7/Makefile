INFO = -Minfo=all
LIBS = -lboost_program_options -cudalib=cublas
GPU = -acc=gpu
CXX = pgc++

all:gpu2
	
gpu2: main2.cpp
	$(CXX) $(GPU) $(INFO) $(LIBS) -o $@ $<

clean:all
	rm gpu2
