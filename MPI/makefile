CC = mpicc
CPPFLAGS = -g
LIBS = -lm
HEADERS = convolution.h mpiFunctions.h
SRC =  main.c convolution.c mpiFunctions.c
OBJS =  main.o convolution.o mpiFunctions.o

all: convolution

%.o: %.c $(HEADERS)
	$(CC) -c $< $ $
	
convolution: $(OBJS)
	$(CC) $(CPPFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	rm -f $(OBJS)