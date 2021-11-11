#!/bin/bash
rm -rf api/* && sphinx-build -b html -E -v . ~/github/HorizonDocs
