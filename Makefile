CC = gcc
CFLAGS = -Wall -Wextra -O2
HEADERS = main.h

# ======================
# Default target
# ======================
all: client server bch_encode bch_decode

# ======================
# Client & Server
# ======================
client: client.o support.o
	$(CC) -o $@ $^

server: server.o support.o
	$(CC) -o $@ $^ -lm

# ======================
# BCH encode/decode
# ======================
bch_encode: bch_encode.c
	$(CC) $(CFLAGS) -o $@ $< -lm -lcrypto

bch_decode: bch_decode.c
	$(CC) $(CFLAGS) -o $@ $< -lm -lcrypto

# ======================
# Object files
# ======================
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $<

# ======================
# Clean
# ======================
clean:
	rm -f *.o client server bch_encode bch_decode
