INFO = -Minfo=all
LIBS = -lboost_program_options
HOST = -acc=host
MULT = -acc=multicore
CXX = pgc++

all:serial multi
	

serial: main.cpp
	$(CXX) $(HOST) $(INFO) $(LIBS) -o $@ $<

multi: main.cpp
	$(CXX) $(MULT) $(INFO) $(LIBS) -o $@ $<

clean:all
	rm serial multi
