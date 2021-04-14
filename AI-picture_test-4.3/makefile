INCDIR  =	-I /usr/local/include/opencv4/  -I /usr/local/include/ 
CFLAGS 	= 	-g  -std=c++11 $(shell pkg-config --libs --static opencv)
SOURCE 	=	 picture_compare.cpp
LD 		=	$(shell pkg-config --libs --static opencv) -lpthread -lcurl
CC 		=	 g++
TARGET 	=	 compare_picture
COBJS   +=	 picture_compare.o
COBJS   +=	 http_updata.o

all: $(TARGET)

$(TARGET) :	$(COBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LD) 
%.o : %.cpp
	$(CC) $(CFLAGS)  -c -o  $@ $^ 
clean:
	rm -f $(TARGET)  *.o *.TMP