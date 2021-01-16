all: 
	@mkdir -p result
	gcc src/utils.c src/network.c src/layers.c src/detect.c -lpthread -lm -Iinclude -o detect
clean:
	rm detect 
