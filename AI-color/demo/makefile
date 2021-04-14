INCDIR  =	-I /usr/local/include/ 
#CFLAGS 	= 	-g  -std=c++11 $(shell pkg-config --libs --static opencv)
CFLAGS 	= 	-g  -std=c++11 $(shell pkg-config --cflags opencv4)
SOURCE 	=	 qrcode.cpp
#LD 		=	$(shell pkg-config --libs --static opencv) -lpthread -lcurl -lzbar
LD 		=	$(shell pkg-config --libs opencv4) -lpthread 
#-lcurl 
CC 		=	 g++
TARGET 	=	 color
COBJS   +=	 color.o

all: $(TARGET)

$(TARGET) :	$(COBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LD) 
%.o : %.cpp
	$(CC) $(CFLAGS)  -c -o  $@ $^ 
clean:
	rm -f $(TARGET)  *.o *.TMP