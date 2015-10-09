#! /bin/bash

gcc -shared -o integrandlib.so -fPIC integration.c -std=gnu99
