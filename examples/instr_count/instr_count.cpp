#include <stdint.h>
#include <stdio.h>

#include "sibir.h"

void sibir_at_init() {
    printf("Initializing\n");
}

void sibir_at_term() {
    printf("Terminating\n");
}

void sibir_at_hipMalloc(void** ptr, size_t size) {
    printf("Allocted %lu bytes\n", size);
}

void instrument_function() {
    printf("We'll be putting some useful code here some day\n");
}